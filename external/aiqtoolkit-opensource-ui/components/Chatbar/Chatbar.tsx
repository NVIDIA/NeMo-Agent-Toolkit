import { useCallback, useContext, useEffect } from 'react';

import { useTranslation } from 'next-i18next';

import { useCreateReducer } from '@/hooks/useCreateReducer';

import { saveConversation, saveConversations } from '@/utils/app/conversation';
import { saveFolders } from '@/utils/app/folders';
import { exportData, importData } from '@/utils/app/importExport';

import { Conversation } from '@/types/chat';
import { LatestExportFormat, SupportedExportFormats } from '@/types/export';

import HomeContext from '@/pages/api/home/home.context';

import { ChatFolders } from './components/ChatFolders';
import { ChatbarSettings } from './components/ChatbarSettings';
import { Conversations } from './components/Conversations';

import Sidebar from '../Sidebar';
import ChatbarContext from './Chatbar.context';
import { ChatbarInitialState, initialState } from './Chatbar.state';

import { v4 as uuidv4 } from 'uuid';

export const Chatbar = () => {
  const { t } = useTranslation('sidebar');

  const chatBarContextValue = useCreateReducer<ChatbarInitialState>({
    initialState,
  });

  const {
    state: { conversations, filteredConversations, showChatbar, folders, currentMode },
    dispatch: homeDispatch,
    handleCreateFolder,
    handleNewConversation,
    handleUpdateConversation,
  } = useContext(HomeContext);

  const {
    state: { searchTerm, filteredConversations: searchFilteredConversations },
    dispatch: chatDispatch,
  } = chatBarContextValue;

  const handleExportData = () => {
    // Import the mode-specific export function
    import('@/utils/app/modeSettings').then(({ exportModeData }) => {
      exportModeData(currentMode);
    });
  };

  const handleImportConversations = (data: SupportedExportFormats) => {
    // Import mode-specific data
    import('@/utils/app/modeSettings').then(({ importModeData }) => {
      try {
        const result = importModeData(data, currentMode);
        
        // Get all conversations from storage (including the newly imported ones)
        const allConversations = JSON.parse(sessionStorage.getItem('conversationHistory') || '[]');
        
        // Update the main conversations array with all conversations
        homeDispatch({ field: 'conversations', value: allConversations });
        
        // Filter conversations for current mode
        const filteredForCurrentMode = allConversations.filter((c: any) => c.mode === currentMode);
        homeDispatch({ field: 'filteredConversations', value: filteredForCurrentMode });
        
        // Select the most recent conversation for this mode
        if (filteredForCurrentMode.length > 0) {
          homeDispatch({
            field: 'selectedConversation',
            value: filteredForCurrentMode[filteredForCurrentMode.length - 1],
          });
        }

        window.location.reload();
      } catch (error) {
        console.error('Import failed:', error);
        // Fallback to original import for backward compatibility
        const { history, folders }: LatestExportFormat = importData(data);
        homeDispatch({ field: 'conversations', value: history });
        homeDispatch({
          field: 'selectedConversation',
          value: history[history.length - 1],
        });
        homeDispatch({ field: 'folders', value: folders });
        window.location.reload();
      }
    });
  };

  const handleClearConversations = () => {
    // Only clear conversations for the current mode
    const conversationsForOtherModes = conversations.filter(c => c.mode !== currentMode);
    
    homeDispatch({
      field: 'selectedConversation',
      value: {
        id: uuidv4(),
        name: t('New Conversation'),
        messages: [],
        folderId: null,
        mode: currentMode,
      },
    });

    // Update conversations to only keep other modes' conversations
    homeDispatch({ field: 'conversations', value: conversationsForOtherModes });
    homeDispatch({ field: 'filteredConversations', value: [] });

    // Update storage - keep other modes' conversations
    saveConversations(conversationsForOtherModes);
    
    // Clear mode-specific storage
    const storageKey = currentMode === 'FRIDAY' ? 'conversationHistory_FRIDAY' : 'conversationHistory_ON_CALL';
    sessionStorage.removeItem(storageKey);
    sessionStorage.removeItem('selectedConversation');

    // Note: We don't clear folders as they might be shared or contain other mode data
  };

  const handleDeleteConversation = (conversation: Conversation) => {
    const updatedConversations = conversations.filter(
      (c) => c.id !== conversation.id,
    );

    homeDispatch({ field: 'conversations', value: updatedConversations });
    chatDispatch({ field: 'searchTerm', value: '' });
    saveConversations(updatedConversations);

    // Update filtered conversations
    const updatedFilteredConversations = filteredConversations.filter(
      (c) => c.id !== conversation.id,
    );
    homeDispatch({ field: 'filteredConversations', value: updatedFilteredConversations });

    if (updatedFilteredConversations.length > 0) {
      homeDispatch({
        field: 'selectedConversation',
        value: updatedFilteredConversations[updatedFilteredConversations.length - 1],
      });

      saveConversation(updatedFilteredConversations[updatedFilteredConversations.length - 1]);
    } else {
        homeDispatch({
          field: 'selectedConversation',
          value: {
            id: uuidv4(),
            name: t('New Conversation'),
            messages: [],
            folderId: null,
            mode: currentMode,
          },
        });

        sessionStorage.removeItem('selectedConversation');
    }
  };

  const handleToggleChatbar = () => {
    homeDispatch({ field: 'showChatbar', value: !showChatbar });
    sessionStorage.setItem('showChatbar', JSON.stringify(!showChatbar));
  };

  const handleDrop = (e: any) => {
    if (e.dataTransfer) {
      const conversation = JSON.parse(e.dataTransfer.getData('conversation'));
      handleUpdateConversation(conversation, { key: 'folderId', value: 0 });
      chatDispatch({ field: 'searchTerm', value: '' });
      e.target.style.background = 'none';
    }
  };

  useEffect(() => {
    if (searchTerm) {
      chatDispatch({
        field: 'filteredConversations',
        value: filteredConversations.filter((conversation) => {
          const searchable =
            conversation.name.toLocaleLowerCase() +
            ' ' +
            conversation.messages.map((message) => message.content).join(' ');
          return searchable.toLowerCase().includes(searchTerm.toLowerCase());
        }),
      });
    } else {
      chatDispatch({
        field: 'filteredConversations',
        value: filteredConversations,
      });
    }
  }, [searchTerm, filteredConversations, chatDispatch]);

  return (
    <ChatbarContext.Provider
      value={{
        ...chatBarContextValue,
        handleDeleteConversation,
        handleClearConversations,
        handleImportConversations,
        handleExportData,
      }}
    >
      <Sidebar<Conversation>
        side={'left'}
        isOpen={showChatbar}
        addItemButtonTitle={t('New chat')}
        itemComponent={<Conversations conversations={searchFilteredConversations} />}
        folderComponent={<ChatFolders searchTerm={searchTerm} />}
        items={searchFilteredConversations}
        searchTerm={searchTerm}
        handleSearchTerm={(searchTerm: string) =>
          chatDispatch({ field: 'searchTerm', value: searchTerm })
        }
        toggleOpen={handleToggleChatbar}
        handleCreateItem={handleNewConversation}
        handleCreateFolder={() => handleCreateFolder(t('New folder'), 'chat')}
        handleDrop={handleDrop}
        footerComponent={<ChatbarSettings />}
      />
    </ChatbarContext.Provider>
  );
};
