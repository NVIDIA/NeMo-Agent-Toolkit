import { Conversation, Message } from '@/types/chat';
import { FolderInterface } from '@/types/folder';
import { t } from 'i18next';

export type AppMode = 'FRIDAY' | 'ON CALL' | 'SLACK';

export interface ModeSpecificSettings {
  chatHistory: boolean;
  chatCompletionURL: string;
  webSocketMode: boolean;
  webSocketConnected: boolean;
  webSocketURL: string;
  webSocketSchema: string;
  enableIntermediateSteps: boolean;
  expandIntermediateSteps: boolean;
  intermediateStepOverride: boolean;
}

export interface HomeInitialState {
  loading: boolean;
  lightMode: 'light' | 'dark'; // Shared between modes
  messageIsStreaming: boolean;
  folders: FolderInterface[];
  conversations: Conversation[];
  filteredConversations: Conversation[];
  selectedConversation: Conversation | undefined;
  currentMessage: Message | undefined;
  showChatbar: boolean;
  currentFolder: FolderInterface | undefined;
  messageError: boolean;
  searchTerm: string;
  // Legacy fields for backward compatibility
  chatHistory: boolean;
  chatCompletionURL?: string;
  webSocketMode?: boolean;
  webSocketConnected?: boolean;
  webSocketURL?: string;
  webSocketSchema?: string;
  webSocketSchemas?: string[];
  enableIntermediateSteps?: boolean;
  expandIntermediateSteps?: boolean;
  intermediateStepOverride?: boolean;
  autoScroll?: boolean;
  additionalConfig: any;
  currentMode: AppMode;
  // Mode-specific settings
  fridaySettings: ModeSpecificSettings;
  onCallSettings: ModeSpecificSettings;
  slackSettings: ModeSpecificSettings;
}

// Get default settings for a specific mode
const getDefaultSettingsForMode = (mode: AppMode): ModeSpecificSettings => {
  const baseSettings = {
    chatHistory: process?.env?.NEXT_PUBLIC_CHAT_HISTORY_DEFAULT_ON === 'true' || true,
    webSocketMode: process?.env?.NEXT_PUBLIC_WEB_SOCKET_DEFAULT_ON === 'true' || false,
    webSocketConnected: false,
    webSocketSchema: 'chat_stream',
    enableIntermediateSteps: process?.env?.NEXT_PUBLIC_ENABLE_INTERMEDIATE_STEPS ? process.env.NEXT_PUBLIC_ENABLE_INTERMEDIATE_STEPS === 'true' : true,
    expandIntermediateSteps: false,
    intermediateStepOverride: true,
  };

  // Mode-specific URLs - Backend ports: FRIDAY=8001, ON_CALL=8000
  if (mode === 'FRIDAY') {
    return {
      ...baseSettings,
      chatCompletionURL: process?.env?.NEXT_PUBLIC_HTTP_CHAT_COMPLETION_URL || 'http://127.0.0.1:8001/chat/stream',
      webSocketURL: process?.env?.NEXT_PUBLIC_WEBSOCKET_CHAT_COMPLETION_URL || 'ws://127.0.0.1:8001/websocket',
    };
  } else if (mode === 'ON CALL') {
    return {
      ...baseSettings,
      chatCompletionURL: process?.env?.NEXT_PUBLIC_HTTP_CHAT_COMPLETION_URL || 'http://127.0.0.1:8000/chat/stream',
      webSocketURL: process?.env?.NEXT_PUBLIC_WEBSOCKET_CHAT_COMPLETION_URL || 'ws://127.0.0.1:8000/websocket',
    };
  } else { // SLACK
    return {
      ...baseSettings,
      chatCompletionURL: process?.env?.NEXT_PUBLIC_HTTP_CHAT_COMPLETION_URL || 'http://127.0.0.1:8002/chat/stream',
      webSocketURL: process?.env?.NEXT_PUBLIC_WEBSOCKET_CHAT_COMPLETION_URL || 'ws://127.0.0.1:8002/websocket',
    };
  }
};

const defaultFridaySettings = getDefaultSettingsForMode('FRIDAY');
const defaultOnCallSettings = getDefaultSettingsForMode('ON CALL');
const defaultSlackSettings = getDefaultSettingsForMode('SLACK');

export const initialState: HomeInitialState = {
  loading: false,
  lightMode: 'light',
  messageIsStreaming: false,
  folders: [],
  conversations: [],
  filteredConversations: [],
  selectedConversation: undefined,
  currentMessage: undefined,
  showChatbar: true,
  currentFolder: undefined,
  messageError: false,
  searchTerm: '',
  // Legacy fields for backward compatibility - default to FRIDAY settings
  chatHistory: defaultFridaySettings.chatHistory,
  chatCompletionURL: defaultFridaySettings.chatCompletionURL,
  webSocketMode: defaultFridaySettings.webSocketMode,
  webSocketConnected: defaultFridaySettings.webSocketConnected,
  webSocketURL: defaultFridaySettings.webSocketURL,
  webSocketSchema: defaultFridaySettings.webSocketSchema,
  webSocketSchemas: ['chat_stream', 'chat', 'generate_stream', 'generate'],
  enableIntermediateSteps: defaultFridaySettings.enableIntermediateSteps,
  expandIntermediateSteps: defaultFridaySettings.expandIntermediateSteps,
  intermediateStepOverride: defaultFridaySettings.intermediateStepOverride,
  autoScroll: true,
  additionalConfig: {},
  currentMode: 'FRIDAY',
  // Mode-specific settings
  fridaySettings: defaultFridaySettings,
  onCallSettings: defaultOnCallSettings,
  slackSettings: defaultSlackSettings,
};
