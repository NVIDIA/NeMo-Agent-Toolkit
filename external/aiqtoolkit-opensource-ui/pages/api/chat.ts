import { ChatBody } from '@/types/chat';
import { delay } from '@/utils/app/helper';
export const config = {
  runtime: 'edge',
  api: {
    bodyParser: {
      sizeLimit: '5mb',
    },
  },
};


const handler = async (req: Request): Promise<Response> => {

  // extract the request body
  let {
    chatCompletionURL = '',
    messages = [],
    additionalProps = {
      enableIntermediateSteps: true
    }
  } = (await req.json()) as ChatBody;

  try {    
    let payload
    // for generate end point the request schema is {input_message: "user question"}
    if(chatCompletionURL.includes('generate')) {
      if (messages?.length > 0 && messages[messages.length - 1]?.role === 'user') {
        payload = {
          input_message: messages[messages.length - 1]?.content ?? ''
        };
      } else {
        throw new Error('User message not found: messages array is empty or invalid.');
      }
    }

    // for chat end point it is openAI compatible schema
    else {
      payload = {
        messages,
        model: "meta/llama-3.3-70b-instruct",
        temperature: 0,
        max_tokens: 4096,
        top_p: 1,
        use_knowledge_base: true,
        top_k: 10,
        collection_name: "string",
        stop: false,
        additionalProp1: {}
      }
    }

    console.log('aiq - making request to', { url: chatCompletionURL });

    let response = await fetch(chatCompletionURL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    console.log('aiq - received response from server', response.status);

    if (!response.ok) {
      let errorMessage = await response.text();

      if(errorMessage.includes('<!DOCTYPE html>')) {
        if(errorMessage.includes('404')) {
          errorMessage = '404 - Page not found'
        }
        else {
          errorMessage = 'HTML response received from server, which cannot be parsed.'
        }
        
      }
      console.log('aiq - received error response from server', errorMessage);
      // For other errors, return a Response object with the error message
      const formattedError = `Something went wrong. Please try again. \n\n<details><summary>Details</summary>Error Message: ${errorMessage || 'Unknown error'}</details>`
      return new Response(formattedError, {
        status: 200, // Return 200 status
        headers: { 'Content-Type': 'text/plain' }, // Set appropriate content type
      });
    }


    // response handling for streaming schema
    if (chatCompletionURL.includes('stream')) {
      console.log('aiq - processing streaming response');
      const encoder = new TextEncoder();
      const decoder = new TextDecoder();

      const responseStream = new ReadableStream({
        async start(controller) {
          const reader = response?.body?.getReader();
          let buffer = '';
          let counter = 0
          try {
            while (true) {
              const { done, value } = await reader?.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(5);
                  if (data.trim() === '[DONE]') {
                    controller.close();
                    return;
                  }
                  try {
                    const parsed = JSON.parse(data);
                    const content = parsed.choices[0]?.message?.content || parsed.choices[0]?.delta?.content || '';
                    if (content) {
                      // console.log(`aiq - stream response received from server with length`, content?.length)
                      controller.enqueue(encoder.encode(content));
                    }
                  } catch (error) {
                    console.log('aiq - error parsing JSON:', error);
                  }
                }
                // TODO - fix or remove this and use websocket to support intermediate data
                if (line.startsWith('intermediate_data: ')) {

                  if(additionalProps.enableIntermediateSteps === true) {
                    const data = line.split('intermediate_data: ')[1];
                    if (data.trim() === '[DONE]') {
                      controller.close();
                      return;
                    }
                    try {
                      const payload = JSON.parse(data);
                      let details = payload?.payload || 'No details';
                      let name = payload?.name || 'Step';
                      let id = payload?.id || '';
                      let status = payload?.status || 'in_progress';
                      let error = payload?.error || '';
                      let type = 'system_intermediate';
                      let parent_id = payload?.parent_id || 'default';
                      let intermediate_parent_id = payload?.intermediate_parent_id || 'default';
                      let time_stamp = payload?.time_stamp || 'default';

                      const intermediate_message = {
                        id,
                        status,
                        error,
                        type,
                        parent_id,
                        intermediate_parent_id,
                        content: {
                          name: name,
                          payload: details,          
                        },
                        time_stamp,
                        index: counter++
                      };
                      const messageString = `<intermediatestep>${JSON.stringify(intermediate_message)}</intermediatestep>`;
                      // console.log('intermediate step counter', counter ++ , messageString.length)
                      controller.enqueue(encoder.encode(messageString));
                      // await delay(1000)
                    } catch (error) {
                      controller.enqueue(encoder.encode('Error parsing intermediate data: ' + error));
                      console.log('aiq - error parsing JSON:', error);
                    }
                  }
                  else {
                    console.log('aiq - intermediate data is not enabled');
                  }
                }
              }
            }
          } catch (error) {
            console.log('aiq - stream reading error, closing stream', error);
            controller.close();
          } finally {
            console.log('aiq - response processing is completed, closing stream');
            controller.close();
            reader?.releaseLock();
          }
        },
      });

      return new Response(responseStream);
    }

    // response handling for non straming schema
    else {
      console.log('aiq - processing non streaming response');
      const data = await response.text();
      let parsed = null;
    
      try {
        parsed = JSON.parse(data);
      } catch (error) {
        console.log('aiq - error parsing JSON response', error);
      }
    
      // Safely extract content with proper checks
      const content =
        parsed?.output || // Check for `output`
        parsed?.answer || // Check for `answer`
        parsed?.value ||  // Check for `value`
        (Array.isArray(parsed?.choices) ? parsed.choices[0]?.message?.content : null) || // Safely check `choices[0]`
        parsed || // Fallback to the entire `parsed` object
        data; // Final fallback to raw `data`
    
      if (content) {
        console.log('aiq - response processing is completed');
        return new Response(content);
      } else {
        console.log('aiq - error parsing response');
        return new Response(response.body || data);
      }
    }
    
  } catch (error) {
    console.log('error - while making request', error);
    const formattedError = `Something went wrong. Please try again. \n\n<details><summary>Details</summary>Error Message: ${error?.message || 'Unknown error'}</details>`
    return new Response(formattedError, { status: 200 })
  }
};

export default handler;
