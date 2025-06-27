import { saveConversation, saveConversations, updateConversation} from '@/utils/app/conversation';
import { v4 as uuidv4 } from 'uuid';

export const useConversationOperations = ({ conversations, dispatch, t, appConfig }) => {
  const handleSelectConversation = (conversation) => {
    dispatch({
      field: 'selectedConversation',
      value: conversation,
    });

    // updating the session id based on the selcted conversation
    sessionStorage.setItem('sessionId', conversation?.id);
    saveConversation(conversation);
  };

  const handleNewConversation = () => {
    const lastConversation = conversations[conversations.length - 1];

    const newConversation = {
      id: uuidv4(),
      name: t('New Conversation'),
      messages: [],
      folderId: null,
    };

    // setting new the session id for new chat conversation
    sessionStorage.setItem('sessionId', newConversation.id);
    const updatedConversations = [...conversations, newConversation];

    dispatch({ field: 'selectedConversation', value: newConversation });
    dispatch({ field: 'conversations', value: updatedConversations });

    saveConversations(updatedConversations);

    dispatch({ field: 'loading', value: false });
  };

  const handleUpdateConversation = (conversation, data) => {
    const updatedConversation = {
      ...conversation,
      [data.key]: data.value,
    };

    const { single, all } = updateConversation(updatedConversation, conversations);

    dispatch({ field: 'selectedConversation', value: single });
    dispatch({ field: 'conversations', value: all });
    
    saveConversations(all);
  };

  return { handleSelectConversation, handleNewConversation, handleUpdateConversation };
};
