import { Conversation } from '@/types/chat';

export const cleanSelectedConversation = (conversation: Conversation) => {
  let updatedConversation = conversation;

  if (!updatedConversation.folderId) {
    updatedConversation = {
      ...updatedConversation,
      folderId: updatedConversation.folderId || null,
    };
  }

  if (!updatedConversation.messages) {
    updatedConversation = {
      ...updatedConversation,
      messages: updatedConversation.messages || [],
    };
  }

  if (conversation.mode) {
    updatedConversation = {
      ...updatedConversation,
      mode: conversation.mode,
    };
  }

  return updatedConversation;
};

export const cleanConversationHistory = (history: any[]): Conversation[] => {

  if (!Array.isArray(history)) {
    console.warn('history is not an array. Returning an empty array.');
    return [];
  }

  return history.reduce((acc: any[], conversation) => {
    try {
      if (!conversation.folderId) {
        conversation.folderId = null;
      }

      if (!conversation.messages) {
        conversation.messages = [];
      }

      const cleanedConversation = {
        ...conversation,
        folderId: conversation.folderId || null,
        messages: conversation.messages || [],
      };

      if (conversation.mode) {
        cleanedConversation.mode = conversation.mode;
      }

      acc.push(cleanedConversation);
      return acc;
    } catch (error) {
      console.warn(
        `error while cleaning conversations' history. Removing culprit`,
        error,
      );
    }
    return acc;
  }, []);
};
