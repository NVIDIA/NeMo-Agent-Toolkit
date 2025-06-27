import {
  IconArrowDown,
  IconBolt,
  IconPaperclip,
  IconPhoto,
  IconPlayerStop,
  IconRepeat,
  IconSend,
  IconTrash,
  IconMicrophone,
  IconPlayerStopFilled,
  IconMicrophone2,
} from '@tabler/icons-react';
import {
  KeyboardEvent,
  MutableRefObject,
  Ref,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';

import { useTranslation } from 'next-i18next';

import { Message } from '@/types/chat';

import HomeContext from '@/pages/api/home/home.context';
import { compressImage, getWorkflowName } from '@/utils/app/helper';
import { appConfig } from '@/utils/app/const';
import toast from 'react-hot-toast';


interface Props {
  onSend: (message: Message) => void;
  onRegenerate: () => void;
  onScrollDownClick: () => void;
  textareaRef: MutableRefObject<HTMLTextAreaElement | null>;
  showScrollDownButton: boolean;
  controller: Ref<AbortController>
}

export const ChatInput = ({
  onSend,
  onRegenerate,
  onScrollDownClick,
  textareaRef,
  showScrollDownButton,
  controller
}: Props) => {
  const { t } = useTranslation('chat');

  const {
    state: { selectedConversation, messageIsStreaming, loading, webSocketMode, currentMode },
    dispatch: homeDispatch,
  } = useContext(HomeContext);

  const workflow = getWorkflowName()

  // todo add the audio file
  const recordingStartSound = new Audio('audio/recording.wav');

  // Hardcoded region configurations for Loki
  const regions = {
    'us-east-1': {
      name: 'US East 1',
      labels: ['cluster', 'clustername'],
      values: {
        cluster: ['aistore-asr-iad', 'alloy-otel-gcp-cbf-cs-001', 'aws-us-east-1-dca-wl-lab-001', 'aws-us-east-1-dca-wl-lab-002', 'aws-us-east-1-dca-wl-lab-003', 'aws-us-east-1-dca-wl-lab-006', 'aws-us-east-2-dca-wl-prd-001', 'dp-riva-oci-prd', 'eks-bridge-prod-east', 'eks-bridge-stg-east', 'gcp-cbf-cs-004', 'gcp-cbf-cs-002', 'gcp-sre-test-001', 'gcp-us-central1-dca-wl-lab-001', 'gcp-us-central1-dca-wl-lab-002', 'gcp-us-central1-dca-wl-lab-003', 'gcp-us-east4-dca-wl-prd-001', 'gnicis-cbg1-jumpbox', 'gnicis-cbg1-stg-jumpbox', 'kratos-multitenant', 'nv-cosmos-storage-prod', 'rg-mc-cnc-ovxcanprdp2-000', 'sre-k8s-dev01', 'sre-k8s-prod01'],
        clustername: ['storage-dfw-c1001', 'storage-dfw-c1002', 'storage-dfw-c1003', 'storage-dfw-c1004', 'storage-dfw-c1005', 'storage-dfw-c1006', 'storage-dfw-c1007', 'storage-dfw-c1008', 'storage-dfw-c1009', 'storage-dfw-c1010', 'storage-dfw-c1011', 'storage-dfw-c1012', 'storage-dfw-c1013', 'storage-dfw-ddn-cwdfwfs1', 'storage-hsg-c501', 'storage-hsg-c502', 'storage-hsg-c503', 'storage-hsg-c504', 'storage-hsg-c505', 'storage-hsg-c506', 'storage-hsg-c507', 'storage-hsg-c508', 'storage-hsg-c509', 'storage-hsg-c510', 'storage-hsg-c511', 'storage-hsg-c512', 'storage-hsg-c513', 'storage-hsg-c514', 'storage-hsg-c515', 'storage-hsg-c516', 'storage-hsg-c517', 'storage-hsg-c518', 'storage-hsg-c519', 'storage-hsg-c520', 'storage-hsg-c521', 'storage-hsg-c522', 'storage-hsg-c523', 'storage-hsg-c524', 'storage-hsg-c525', 'storage-hsg-c526', 'storage-hsg-c527', 'storage-hsg-c528', 'storage-hsg-c529', 'storage-hsg-c530', 'storage-hsg-c531', 'storage-hsg-c532', 'storage-hsg-c533', 'storage-hsg-c534', 'storage-hsg-c535', 'storage-hsg-c536']
      }
    },
    'us-west-2': {
      name: 'US West 2',
      labels: ['service', 'level', 'app', 'container'],
      values: {
        service: ['api-gateway', 'data-service', 'cache-service'],
        level: ['ERROR', 'WARN', 'INFO', 'DEBUG'],
        app: ['python3', 'nginx', 'redis'],
        container: ['web-container', 'db-container', 'cache-container']
      }
    },
    'eu-north-1': {
      name: 'EU North 1',
      labels: ['service', 'level', 'namespace', 'system_type'],
      values: {
        service: ['frontend', 'backend', 'database'],
        level: ['ERROR', 'WARN', 'INFO', 'DEBUG'],
        namespace: ['production', 'staging', 'development'],
        system_type: ['gb200', 'cpu-intensive', 'memory-optimized']
      }
    },
    'ap-northeast-1': {
      name: 'AP Northeast 1',
      labels: ['service', 'level', 'cluster', 'app'],
      values: {
        service: ['order-service', 'inventory-service', 'shipping-service'],
        level: ['ERROR', 'WARN', 'INFO', 'DEBUG'],
        cluster: ['kratos-multitenant', 'asia-prod-cluster'],
        app: ['sshd', 'systemd', 'docker']
      }
    }
  };

  const [content, setContent] = useState<string>('');
  const [isTyping, setIsTyping] = useState<boolean>(false);
  const fileInputRef = useRef(null);
  const [inputFile, setInputFile] = useState(null)
  const [inputFileExtension, setInputFileExtension] = useState('')
  const [inputFileContent, setInputFileContent] = useState('')
  const [inputFileContentCompressed, setInputFileContentCompressed] = useState('')
  const [isRecording, setIsRecording] = useState(false);
  const recognitionRef = useRef(null);
  // Loki query builder states
  const [showLokiDropdowns, setShowLokiDropdowns] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState('');
  const [selectedLabel, setSelectedLabel] = useState('');
  const [selectedValue, setSelectedValue] = useState('');

  const triggerFileUpload = () => {
    fileInputRef?.current.click();
  };

  const handleInputFileDelete = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputFile(null);
    setInputFileExtension('');
    setInputFileContent('');
    setInputFileContentCompressed('');
  };

  const handleFileChange = (e: { target: { files: any[]; value: null; }; }) => {
    const file = e.target.files[0]
    if (file) {
      // Reset the input value so the same file can be selected again if needed
      e.target.value = null
      const reader = new FileReader()
      reader.onload = (loadEvent) => {
        const fullBase64String = loadEvent.target?.result;
        processFile({ fullBase64String, file })
      };
      reader.readAsDataURL(file)
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;

    setContent(value);
    
    // Simple if-else logic to detect Loki queries in Friday mode
    if (currentMode === 'FRIDAY') {
      const isLokiQuery = value.toLowerCase().includes('loki') || 
                         value.toLowerCase().includes('logs') || 
                         value.toLowerCase().includes('log analysis');
      setShowLokiDropdowns(isLokiQuery);
      
      // Reset selections when query changes
      if (!isLokiQuery) {
        setSelectedRegion('');
        setSelectedLabel('');
        setSelectedValue('');
      }
    } else {
      setShowLokiDropdowns(false);
    }
  };

  const handleSend = () => {
    if (messageIsStreaming) {
      return;
    }

    // stop recognition if it's running
    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    }

    if (!content.trim() && !inputFile && !inputFileContent) {
      toast.error(t('Please enter a message'));
      return;
    }

    if (inputFile || inputFileContent) {
      onSend({
        role: 'user',
        content: content,
        attachments: [{
          content: inputFileContent,
          type: 'image'
        }]
      })
      setContent('');
      setInputFile(null)
      setInputFileExtension('')
      setInputFileContent('')
      setInputFileContentCompressed('');
    }

    else {
      onSend({ role: 'user', content })
      setContent('');
      setInputFile(null)
      setInputFileExtension('')
      setInputFileContent('')
      setInputFileContentCompressed('');
    }


    if (window.innerWidth < 640 && textareaRef && textareaRef.current) {
      textareaRef.current.blur();
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !isTyping && !isMobile() && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    } else if (e.key === '/' && e.metaKey) {
      e.preventDefault();
    }
  };


  const handleStopConversation = () => {
    if(webSocketMode) {
      console.log('stop conversation is yet to be implemented');
      homeDispatch({ field: 'loading', value: false });
      homeDispatch({ field: 'messageIsStreaming', value: false });
      return 
    }
    else {
      try {
        controller?.current?.abort('aborted');
        setTimeout(() => {
          controller.current = new AbortController(); // Reset the controller
        }, 100);
      } catch (error) {
        console.log('error aborting - ', error);
      }
    }
  };

  const isMobile = () => {
    const userAgent =
      typeof window.navigator === 'undefined' ? '' : navigator.userAgent;
    const mobileRegex =
      /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|mobile|CriOS/i;
    return mobileRegex.test(userAgent);
  };

  // Simple handlers for Loki dropdowns
  const handleRegionChange = (region: string) => {
    setSelectedRegion(region);
    setSelectedLabel(''); // Reset label when region changes
    setSelectedValue(''); // Reset value when region changes
  };

  const handleLabelChange = (label: string) => {
    setSelectedLabel(label);
    setSelectedValue(''); // Reset value when label changes
  };

  const handleValueChange = (value: string) => {
    setSelectedValue(value);
  };

  const applyLokiFilters = () => {
    let updatedQuery = content;
    if (selectedRegion && selectedLabel && selectedValue) {
      updatedQuery += ` [Region: ${selectedRegion}, ${selectedLabel}: ${selectedValue}]`;
      setContent(updatedQuery);
      setShowLokiDropdowns(false);
      // Reset selections
      setSelectedRegion('');
      setSelectedLabel('');
      setSelectedValue('');
    }
  };


  const processFile = ({ fullBase64String, file }: { fullBase64String: string, file: File }) => {
    const [fileType] = file && file.type.split('/');
    if (!["image"].includes(fileType)) {
      alert(`Only supported file types are : ${["image"].join(', ')}`);
      return;
    }

    if (file && file.size > 2 * 1024 * 1024) {
      alert(`File size should not exceed : 2 MB`);
      return;
    }

    const base64WithoutPrefix = fullBase64String.replace(/^data:image\/[a-z]+;base64,/, '');
    const sizeInKB = (base64WithoutPrefix.length * 3 / 4) / 1024;
    // Compress image only if it larger than 200KB
    const shouldCompress = sizeInKB > 200;

    if (shouldCompress) {
      compressImage(fullBase64String, file.type, true, (compressedBase64: string) => {
        setInputFileContentCompressed(compressedBase64);
        setInputFileContent(fullBase64String);
        setInputFile(file?.name);
        const extension = file.name.split('.').pop() ?? 'jpg';
        setInputFileExtension(extension.toLowerCase());
      });
    } else {
      // If no compression is needed, use the original image data
      setInputFileContent(fullBase64String);
      setInputFileContentCompressed(fullBase64String);
      setInputFile(file.name);
      const extension = file.name.split('.').pop() ?? 'jpg';
      setInputFileExtension(extension.toLowerCase());
    }
  }


  const handleInitModal = () => {
    const selectedPrompt = filteredPrompts[activePromptIndex];
    if (selectedPrompt) {
      setContent((prevContent) => {
        const newContent = prevContent?.replace(
          /\/\w*$/,
          selectedPrompt.content,
        );
        return newContent;
      });
      handlePromptSelect(selectedPrompt);
    }
    setShowPromptList(false);
  };

  const parseVariables = (content: string) => {
    const regex = /{{(.*?)}}/g;
    const foundVariables = [];
    let match;

    while ((match = regex.exec(content)) !== null) {
      foundVariables.push(match[1]);
    }

    return foundVariables;
  };

  const handleSubmit = (updatedVariables: string[]) => {
    const newContent = content?.replace(/{{(.*?)}}/g, (match, variable) => {
      const index = variables.indexOf(variable);
      return updatedVariables[index];
    });

    setContent(newContent);

    if (textareaRef && textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // Additional handlers for drag and drop
  const handleDragOver = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault(); // Necessary to allow the drop event
  };

  const handleDrop = (e: React.DragEvent<HTMLTextAreaElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const reader = new FileReader();
      reader.onload = (loadEvent) => {
        const fullBase64String = loadEvent.target?.result;
        processFile({ fullBase64String, file })
      };
      reader.readAsDataURL(file);

    }
  };

  const handlePaste = (event: { clipboardData: any; originalEvent: { clipboardData: any; }; }) => {
    const clipboardData = event.clipboardData || event.originalEvent.clipboardData;
    let items = clipboardData.items;
    let isImagePasted = false;

    if (items) {
      for (const item of items) {
        if (item.type.indexOf("image") === 0) {
          isImagePasted = true;
          const file = item.getAsFile();
          // Reading the image as Data URL (base64)
          const reader = new FileReader();
          reader.onload = (loadEvent) => {
            const fullBase64String = loadEvent.target?.result;
            processFile({ fullBase64String, file })
          };
          reader.readAsDataURL(file);
          break; // Stop checking after finding image, preventing any text setting
        }
      }
    }

    // Handle text only if no image was pasted
    if (!isImagePasted) {
      let text = clipboardData.getData('text/plain');
      if (text) {
        // setContent(text); // Set text content only if text is pasted
      }
    }
  };

  useEffect(() => {
    if (textareaRef && textareaRef.current) {
      textareaRef.current.style.height = 'inherit';
      textareaRef.current.style.height = `${textareaRef.current?.scrollHeight}px`;
      textareaRef.current.style.overflow = `${textareaRef?.current?.scrollHeight > 400 ? 'auto' : 'hidden'
        }`;
    }
  }, [content, textareaRef]);

  const handleSpeechToText = useCallback(() => {
    if (!recognitionRef.current) {
      const SpeechRecognition =
        window?.SpeechRecognition || window?.webkitSpeechRecognition;

      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.lang = 'en-US';
      recognitionRef.current.interimResults = true;
      recognitionRef.current.continuous = true;

      recognitionRef.current.onresult = (event) => {
        let currentTranscript = '';
        for (let i = 0; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setContent(currentTranscript);
      };

      recognitionRef.current.onend = () => {
        if (isRecording) {
          recognitionRef.current.start();
        }
      };
    }

    if (!isRecording) {
      // Play sound when recording starts
      recordingStartSound.play();
      recognitionRef.current.start();
      setIsRecording(true);
    } else {
      recognitionRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Get placeholder text based on current mode
  const getPlaceholderText = () => {
    if (isRecording) return 'Listening...';
    if (currentMode === 'ON CALL') return 'Paste the pagerduty link here';
    return `Unlock ${currentMode} knowledge and expertise`;
  };

  return (
    <div className={`absolute bottom-0 left-0 w-full border-transparent bg-gradient-to-b from-transparent via-white to-white pt-6 dark:border-white/20 dark:via-[#343541] dark:to-[#343541] ${isMobile() ? 'pb-14' : 'pb-4'}`}>
      <div className="stretch mx-2 mt-4 flex flex-row gap-3 last:mb-2 md:mx-4 md:mt-[52px] 2xl:mx-auto 2xl:max-w-5xl">
        {messageIsStreaming && (
          <button
            className="absolute top-0 left-0 right-0 mx-auto mb-3 flex w-fit items-center gap-3 rounded border border-neutral-200 bg-white py-2 px-4 text-black hover:opacity-50 dark:border-neutral-600 dark:bg-[#343541] dark:text-white md:mb-0 md:mt-2"
            onClick={handleStopConversation}
          >
            <IconPlayerStop size={16} /> {t('Stop Generating')}
          </button>
        )}

        {!messageIsStreaming &&
          selectedConversation &&
          selectedConversation.messages.length > 1 &&
          // selectedConversation.messages[selectedConversation.messages.length - 1].role === 'assistant' && 
          (
            <button
              className="absolute top-0 left-0 right-0 mx-auto mb-3 flex w-fit items-center gap-3 rounded border border-neutral-200 bg-white py-2 px-4 text-black hover:opacity-50 dark:border-neutral-600 dark:bg-[#343541] dark:text-white md:mb-0 md:mt-2"
              onClick={onRegenerate}
            >
              <IconRepeat size={16} /> {t('Regenerate response')}
            </button>
          )}

        <div className="relative mx-2 flex w-full flex-grow flex-col rounded-md border border-black/10 bg-white shadow-[0_0_10px_rgba(0,0,0,0.10)] dark:border-gray-900/50 dark:bg-[#40414F] dark:text-white dark:shadow-[0_0_15px_rgba(0,0,0,0.10)] sm:mx-4">
          <textarea
            ref={textareaRef}
            className="m-0 w-full resize-none border-0 sm:p-3 sm:pl-8 bg-transparent p-0 py-2 pr-8 pl-10 text-black dark:bg-transparent dark:text-white md:py-3 md:pl-10 outline-none"
            style={{
              resize: 'none',
              bottom: `${textareaRef?.current?.scrollHeight}px`,
              minHeight: '44px',
              maxHeight: '400px',
              overflow: `${textareaRef.current && textareaRef.current.scrollHeight > 400 ? 'auto' : 'hidden'}`,
            }}
            placeholder={getPlaceholderText()}
            value={content}
            rows={1}
            onCompositionStart={() => setIsTyping(true)}
            onCompositionEnd={() => setIsTyping(false)}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            {...(appConfig?.fileUploadEnabled && {
              onDragOver: handleDragOver,
              onDrop: handleDrop,
              onPaste: handlePaste
            })}
          />
          {inputFile && inputFileContent &&
            <div>
              <div className="relative right-0 top-0 p-1 bg-[#91c438] dark:bg-green-700 text-black dark:text-white flex items-center justify-start gap-2 rounded-small">
                <IconPhoto
                  className="ml-8"
                  size={16}
                />
                <span >{inputFile}</span>
                <IconTrash
                  className="hover:text-[#ff1717e9] cursor-pointer"
                  size={16}
                  onClick={handleInputFileDelete}
                />
              </div>
            </div>
          }

          {/* Simple Loki Dropdowns - only show if Loki query detected in Friday mode */}
          {showLokiDropdowns && currentMode === 'FRIDAY' && (
            <div 
              className="absolute bottom-full left-0 right-0 mb-2 bg-white dark:bg-[#40414F] border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg p-4 z-40 pointer-events-auto"
              style={{ pointerEvents: 'auto' }}
              onWheel={(e) => e.stopPropagation()}
              onTouchMove={(e) => e.stopPropagation()}
            >
              <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">Loki Query Filters</h4>
              
              {/* Region Dropdown */}
              <div className="mb-3">
                <label className="block text-xs text-gray-700 dark:text-gray-300 mb-1">Region:</label>
                <select 
                  value={selectedRegion} 
                  onChange={(e) => handleRegionChange(e.target.value)}
                  className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-[#343541] text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select Region...</option>
                  {Object.keys(regions).map(regionKey => (
                    <option key={regionKey} value={regionKey}>
                      {regions[regionKey as keyof typeof regions].name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Label Dropdown - only show if region selected */}
              {selectedRegion && (
                <div className="mb-3">
                  <label className="block text-xs text-gray-700 dark:text-gray-300 mb-1">Label:</label>
                  <select 
                    value={selectedLabel} 
                    onChange={(e) => handleLabelChange(e.target.value)}
                    className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-[#343541] text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Label...</option>
                    {regions[selectedRegion as keyof typeof regions].labels.map(label => (
                      <option key={label} value={label}>{label}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Value Dropdown - only show if region and label selected */}
              {selectedRegion && selectedLabel && (
                <div className="mb-3">
                  <label className="block text-xs text-gray-700 dark:text-gray-300 mb-1">Value:</label>
                  <select 
                    value={selectedValue} 
                    onChange={(e) => handleValueChange(e.target.value)}
                    className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-[#343541] text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select Value...</option>
                    {regions[selectedRegion as keyof typeof regions].values[selectedLabel as keyof typeof regions[typeof selectedRegion]['values']]?.map(value => (
                      <option key={value} value={value}>{value}</option>
                    ))}
                  </select>
                </div>
              )}

              {/* Apply Button - only show if all selections made */}
              {selectedRegion && selectedLabel && selectedValue && (
                <div className="flex justify-end">
                  <button 
                    onClick={applyLokiFilters}
                    className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    Apply Filters
                  </button>
                </div>
              )}
            </div>
          )}
          {
            appConfig?.fileUploadEnabled && !inputFile &&
            <>
              <button
                className="absolute right-10 top-2 rounded-sm p-1 text-neutral-800 opacity-60 hover:text-[#76b900] dark:bg-opacity-50 dark:text-neutral-100 dark:hover:text-neutral-200"
                onClick={triggerFileUpload}
              >
                {messageIsStreaming ? (
                  <></>
                ) : (
                  <>
                    <IconPaperclip size={18} />
                  </>
                )}
              </button>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </>
          }
          {currentMode !== 'ON CALL' && (
            <button
              onClick={handleSpeechToText}
              className={`absolute left-2 top-2 rounded-sm p-[5px] text-neutral-800 opacity-60 dark:bg-opacity-50 dark:text-neutral-100 ${messageIsStreaming
                ? 'text-neutral-400' // Disable hover and change color when streaming
                : 'hover:text-[#76b900] dark:hover:text-neutral-200' // Normal hover effect
                }`}
              disabled={messageIsStreaming}
            >
              {isRecording ? (
                <IconPlayerStopFilled size={18} className="text-red-500 animate-blink" />
              ) : (
                <IconMicrophone size={18} />
              )}
            </button>
          )}
          <button
            className="absolute right-2 top-2 rounded-sm p-1 text-neutral-800 opacity-60 hover:bg-neutral-200 hover:text-neutral-900 dark:bg-opacity-50 dark:text-neutral-100 dark:hover:text-neutral-200"
            onClick={handleSend}
          >
            {messageIsStreaming ? (
              <div className="h-4 w-4 animate-spin rounded-full border-t-2 border-neutral-800 opacity-60 dark:border-neutral-100"></div>
            ) : (
              <IconSend size={18} />
            )}
          </button>

          {showScrollDownButton && (
            <div className="absolute bottom-12 right-0 lg:bottom-2 lg:-right-10">
              <button
                className="flex h-7 w-7 items-center justify-center rounded-full bg-neutral-300 text-gray-800 shadow-md hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-neutral-200"
                onClick={onScrollDownClick}
              >
                <IconArrowDown size={18} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
