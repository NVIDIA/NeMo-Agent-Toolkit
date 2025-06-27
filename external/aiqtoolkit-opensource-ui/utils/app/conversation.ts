import { Conversation, Role } from '@/types/chat';
import toast from 'react-hot-toast';

export const updateConversation = (
  updatedConversation: Conversation,
  allConversations: Conversation[],
) => {
  const updatedConversations = allConversations.map((c) => {
    if (c.id === updatedConversation.id) {
      return updatedConversation;
    }

    return c;
  });

  saveConversation(updatedConversation);
  saveConversations(updatedConversations);

  return {
    single: updatedConversation,
    all: updatedConversations,
  };
};

export const saveConversation = (conversation: Conversation) => {
  try {
    sessionStorage.setItem('selectedConversation', JSON.stringify(conversation));
  } catch (error) {
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      console.log('Storage quota exceeded, cannot save conversation.');
      toast.error('Storage quota exceeded, cannot save conversation.');
    }
  }
};

export const saveConversations = (conversations: Conversation[]) => {
  try {
    sessionStorage.setItem('conversationHistory', JSON.stringify(conversations));
    
    // Also save conversations separately by mode for backup
    const fridayConversations = conversations.filter(c => c.mode === 'FRIDAY');
    const onCallConversations = conversations.filter(c => c.mode === 'ON CALL');
    
    sessionStorage.setItem('conversationHistory_FRIDAY', JSON.stringify(fridayConversations));
    sessionStorage.setItem('conversationHistory_ON_CALL', JSON.stringify(onCallConversations));
  } catch (error) {
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      console.log('Storage quota exceeded, cannot save conversations.');
      toast.error('Storage quota exceeded, cannot save conversation.');
    }
  }
};

export const loadConversations = () => {
  try {
    const conversationHistory = sessionStorage.getItem('conversationHistory');
    if (conversationHistory) {
      const parsed = JSON.parse(conversationHistory);
      return parsed;
    }
    
    // Fallback: try to load from mode-specific storage
    const fridayConversations = sessionStorage.getItem('conversationHistory_FRIDAY');
    const onCallConversations = sessionStorage.getItem('conversationHistory_ON_CALL');
    
    const allConversations = [
      ...(fridayConversations ? JSON.parse(fridayConversations) : []),
      ...(onCallConversations ? JSON.parse(onCallConversations) : [])
    ];
    
    return allConversations;
  } catch (error) {
    console.error('Error loading conversations:', error);
    return [];
  }
};

