'use client'
import { useEffect, useRef } from 'react';

import { GetServerSideProps } from 'next';
import { useTranslation } from 'next-i18next';
import { serverSideTranslations } from 'next-i18next/serverSideTranslations';
import Head from 'next/head';

import { useCreateReducer } from '@/hooks/useCreateReducer';


import {
  cleanConversationHistory,
  cleanSelectedConversation,
} from '@/utils/app/clean';
import {
  saveConversation,
  saveConversations,
  updateConversation,
  loadConversations,
} from '@/utils/app/conversation';
import { saveFolders } from '@/utils/app/folders';
import { getSettings } from '@/utils/app/settings';
import { getModeSettings, saveModeSettings, updateModeSetting } from '@/utils/app/modeSettings';

import { Conversation } from '@/types/chat';
import { KeyValuePair } from '@/types/data';
import { FolderInterface, FolderType } from '@/types/folder';

import { Chat } from '@/components/Chat/Chat';
import { Chatbar } from '@/components/Chatbar/Chatbar';
import { Navbar } from '@/components/Mobile/Navbar';

import HomeContext from './home.context';
import { HomeInitialState, initialState, AppMode } from './home.state';

import { v4 as uuidv4 } from 'uuid';
import { getWorkflowName } from '@/utils/app/helper';

const Home = (props: any) => {
  const { t } = useTranslation('chat');

  const contextValue = useCreateReducer<HomeInitialState>({
    initialState,
  });

  const {
    state: {
      lightMode,
      folders,
      conversations,
      filteredConversations,
      selectedConversation,
      currentMode,
    },
    dispatch,
  } = contextValue;

  const stopConversationRef = useRef<boolean>(false);

  // Filter conversations based on current mode
  const filterConversationsByMode = (conversations: Conversation[], mode: AppMode) => {
    return conversations.filter(conv => conv.mode === mode);
  };

  // Sync mode-specific settings to legacy fields for backward compatibility
  const syncModeSettingsToLegacy = (mode: AppMode) => {
    const modeSettings = getModeSettings(mode);
    dispatch({ field: 'chatHistory', value: modeSettings.chatHistory });
    dispatch({ field: 'chatCompletionURL', value: modeSettings.chatCompletionURL });
    dispatch({ field: 'webSocketMode', value: modeSettings.webSocketMode });
    dispatch({ field: 'webSocketConnected', value: modeSettings.webSocketConnected });
    dispatch({ field: 'webSocketURL', value: modeSettings.webSocketURL });
    dispatch({ field: 'webSocketSchema', value: modeSettings.webSocketSchema });
    dispatch({ field: 'enableIntermediateSteps', value: modeSettings.enableIntermediateSteps });
    dispatch({ field: 'expandIntermediateSteps', value: modeSettings.expandIntermediateSteps });
    dispatch({ field: 'intermediateStepOverride', value: modeSettings.intermediateStepOverride });
  };

  const handleSelectConversation = (conversation: Conversation) => {
    dispatch({
      field: 'selectedConversation',
      value: conversation,
    });

    saveConversation(conversation);
  };

  // FOLDER OPERATIONS  --------------------------------------------

  const handleCreateFolder = (name: string, type: FolderType) => {
    const newFolder: FolderInterface = {
      id: uuidv4(),
      name,
      type,
    };

    const updatedFolders = [...folders, newFolder];

    dispatch({ field: 'folders', value: updatedFolders });
    saveFolders(updatedFolders);
  };

  const handleDeleteFolder = (folderId: string) => {
    const updatedFolders = folders.filter((f) => f.id !== folderId);
    dispatch({ field: 'folders', value: updatedFolders });
    saveFolders(updatedFolders);

    const updatedConversations: Conversation[] = conversations.map((c) => {
      if (c.folderId === folderId) {
        return {
          ...c,
          folderId: null,
        };
      }

      return c;
    });

    dispatch({ field: 'conversations', value: updatedConversations });
    saveConversations(updatedConversations);
  };

  const handleUpdateFolder = (folderId: string, name: string) => {
    const updatedFolders = folders.map((f) => {
      if (f.id === folderId) {
        return {
          ...f,
          name,
        };
      }

      return f;
    });

    dispatch({ field: 'folders', value: updatedFolders });

    saveFolders(updatedFolders);
  };

  // CONVERSATION OPERATIONS  --------------------------------------------

  const handleNewConversation = () => {
    const newConversation: Conversation = {
      id: uuidv4(),
      name: t('New Conversation'),
      messages: [],
      folderId: null,
      mode: currentMode,
    };

    const updatedConversations = [...conversations, newConversation];

    dispatch({ field: 'selectedConversation', value: newConversation });
    dispatch({ field: 'conversations', value: updatedConversations });

    // Update filtered conversations for current mode
    const filtered = filterConversationsByMode(updatedConversations, currentMode);
    dispatch({ field: 'filteredConversations', value: filtered });

    saveConversation(newConversation);
    saveConversations(updatedConversations);

    dispatch({ field: 'loading', value: false });
  };

  const handleUpdateConversation = (
    conversation: Conversation,
    data: KeyValuePair,
  ) => {
    const updatedConversation = {
      ...conversation,
      [data.key]: data.value,
    };

    const { single, all } = updateConversation(
      updatedConversation,
      conversations,
    );

    dispatch({ field: 'selectedConversation', value: single });
    dispatch({ field: 'conversations', value: all });

    // Update filtered conversations for current mode
    const filtered = filterConversationsByMode(all, currentMode);
    dispatch({ field: 'filteredConversations', value: filtered });
  };

  // EFFECTS  --------------------------------------------

  useEffect(() => {
    if (window.innerWidth < 640) {
      dispatch({ field: 'showChatbar', value: false });
    }
  }, [selectedConversation, dispatch]);

  // Effect to filter conversations when mode changes
  useEffect(() => {
    const filtered = filterConversationsByMode(conversations, currentMode);
    dispatch({ field: 'filteredConversations', value: filtered });

    // Sync mode-specific settings to legacy fields
    syncModeSettingsToLegacy(currentMode);

    // If current selected conversation doesn't belong to current mode, create a new one
    if (selectedConversation && selectedConversation.mode !== currentMode) {
      const modeConversations = filtered;
      if (modeConversations.length > 0) {
        // Select the most recent conversation for this mode
        const mostRecentConversation = modeConversations[modeConversations.length - 1];
        dispatch({ field: 'selectedConversation', value: mostRecentConversation });
        saveConversation(mostRecentConversation);
      } else {
        // Create a new conversation for this mode
        const newConversation: Conversation = {
          id: uuidv4(),
          name: t('New Conversation'),
          messages: [],
          folderId: null,
          mode: currentMode,
        };
        
        // Add the new conversation to the conversations array
        const updatedConversations = [...conversations, newConversation];
        dispatch({ field: 'conversations', value: updatedConversations });
        dispatch({ field: 'selectedConversation', value: newConversation });
        
        // Update filtered conversations
        const updatedFiltered = filterConversationsByMode(updatedConversations, currentMode);
        dispatch({ field: 'filteredConversations', value: updatedFiltered });
        
        // Save the new conversation
        saveConversation(newConversation);
        saveConversations(updatedConversations);
      }
    }
  }, [currentMode, conversations, selectedConversation, dispatch, t]);

  useEffect(() => {
    const settings = getSettings();
    if (settings.theme) {
      dispatch({
        field: 'lightMode',
        value: settings.theme,
      });
    }

    // Load current mode from sessionStorage
    const storedMode = sessionStorage.getItem('currentMode') as AppMode;
    const detectedMode = (storedMode && (storedMode === 'FRIDAY' || storedMode === 'ON CALL')) ? storedMode : 'FRIDAY';
    
    // Set the detected mode
    dispatch({
      field: 'currentMode',
      value: detectedMode,
    });
    
    // Store the detected mode
    sessionStorage.setItem('currentMode', detectedMode);

    // Load mode-specific settings
    const fridaySettings = getModeSettings('FRIDAY');
    const onCallSettings = getModeSettings('ON CALL');
    dispatch({ field: 'fridaySettings', value: fridaySettings });
    dispatch({ field: 'onCallSettings', value: onCallSettings });

    // Sync current mode settings to legacy fields
    syncModeSettingsToLegacy(detectedMode);

    const showChatbar = sessionStorage.getItem('showChatbar');
    if (showChatbar) {
      dispatch({ field: 'showChatbar', value: showChatbar === 'true' });
    }

    const folders = sessionStorage.getItem('folders');
    if (folders) {
      dispatch({ field: 'folders', value: JSON.parse(folders) });
    }

    const loadedConversations = loadConversations();
    if (loadedConversations.length > 0) {
      const cleanedConversationHistory = cleanConversationHistory(loadedConversations);

      // Only migrate truly old conversations without mode (preserve existing modes)
      const migratedConversations = cleanedConversationHistory.map(conv => {
        // If conversation already has a valid mode, keep it
        if (conv.mode && (conv.mode === 'FRIDAY' || conv.mode === 'ON CALL')) {
          return conv;
        }
        // Only apply default mode to conversations without any mode
        return {
          ...conv,
          mode: 'FRIDAY' as AppMode
        };
      });

      dispatch({ field: 'conversations', value: migratedConversations });
      
      // Filter conversations for current mode
      const filtered = filterConversationsByMode(migratedConversations, detectedMode);
      dispatch({ field: 'filteredConversations', value: filtered });
    }

    const selectedConversation = sessionStorage.getItem('selectedConversation');
    if (selectedConversation) {
      const parsedSelectedConversation: Conversation =
        JSON.parse(selectedConversation);
      const cleanedSelectedConversation = cleanSelectedConversation(
        parsedSelectedConversation,
      );

      // Ensure selected conversation has a mode (preserve existing valid modes)
      const conversationWithMode = {
        ...cleanedSelectedConversation,
        mode: (cleanedSelectedConversation.mode && (cleanedSelectedConversation.mode === 'FRIDAY' || cleanedSelectedConversation.mode === 'ON CALL')) 
          ? cleanedSelectedConversation.mode 
          : 'FRIDAY' as AppMode
      };

      dispatch({
        field: 'selectedConversation',
        value: conversationWithMode,
      });
    } else {
      dispatch({
        field: 'selectedConversation',
        value: {
          id: uuidv4(),
          name: t('New Conversation'),
          messages: [],
          folderId: null,
          mode: detectedMode,
        },
      });
    }
  }, [dispatch, t]);

  return (
    <HomeContext.Provider
      value={{
        ...contextValue,
        handleNewConversation,
        handleCreateFolder,
        handleDeleteFolder,
        handleUpdateFolder,
        handleSelectConversation,
        handleUpdateConversation,
      }}
    >
      <Head>
        <title>{currentMode}</title>
        <meta name="description" content={currentMode} />
        <meta
          name="viewport"
          content="height=device-height ,width=device-width, initial-scale=1, user-scalable=no"
        />
        <link rel="icon" href="/nvidia.jpg" />
      </Head>
      {selectedConversation && (
        <main
          className={`flex h-screen w-screen flex-col text-sm text-white dark:text-white ${lightMode}`}
        >
          <div className="fixed top-0 w-full sm:hidden">
            <Navbar
              selectedConversation={selectedConversation}
              onNewConversation={handleNewConversation}
            />
          </div>

          <div className="flex h-full w-full sm:pt-0">
            <Chatbar />

            <div className="flex flex-1">
              <Chat />
            </div>
          </div>
        </main>
      )}
    </HomeContext.Provider>
  );
};
export default Home;

export const getServerSideProps: GetServerSideProps = async ({ locale }) => {
  const defaultModelId = 
  process.env.DEFAULT_MODEL || '';

  return {
    props: {
      defaultModelId,
      ...(await serverSideTranslations(locale ?? 'en', [
        'common',
        'chat',
        'sidebar',
        'markdown',
        'promptbar',
        'settings',
      ])),
    },
  };
};
