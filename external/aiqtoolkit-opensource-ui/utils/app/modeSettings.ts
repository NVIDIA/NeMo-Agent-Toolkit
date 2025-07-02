import { AppMode, ModeSpecificSettings } from '@/pages/api/home/home.state';

// Get the current settings for a specific mode
export const getModeSettings = (mode: AppMode): ModeSpecificSettings => {
  let storageKey;
  if (mode === 'FRIDAY') storageKey = 'fridaySettings';
  else if (mode === 'ON CALL') storageKey = 'onCallSettings';
  else storageKey = 'slackSettings';
  const stored = sessionStorage.getItem(storageKey);
  
  if (stored) {
    try {
      return JSON.parse(stored);
    } catch (error) {
      console.error(`Error parsing ${mode} settings:`, error);
    }
  }
  
  // Return mode-specific default settings
  const defaultSettings = getDefaultSettingsForMode(mode);
  return defaultSettings;
};

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

  // Mode-specific URLs - Use environment variables for Docker deployment
  if (mode === 'FRIDAY') {
    const baseURL = process?.env?.NEXT_PUBLIC_FRIDAY_BACKEND_URL || 'http://127.0.0.1:8001';
    return {
      ...baseSettings,
      chatCompletionURL: `${baseURL}/chat/stream`,
      webSocketURL: baseURL.replace('http', 'ws') + '/websocket',
    };
  } else if (mode === 'ON CALL') {
    const baseURL = process?.env?.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';
    return {
      ...baseSettings,
      chatCompletionURL: `${baseURL}/chat/stream`,
      webSocketURL: baseURL.replace('http', 'ws') + '/websocket',
    };
  } else { // SLACK
    const baseURL = process?.env?.NEXT_PUBLIC_SLACK_BACKEND_URL || 'http://127.0.0.1:8002';
    return {
      ...baseSettings,
      chatCompletionURL: `${baseURL}/chat/stream`,
      webSocketURL: baseURL.replace('http', 'ws') + '/websocket',
    };
  }
};

// Save settings for a specific mode
export const saveModeSettings = (mode: AppMode, settings: ModeSpecificSettings): void => {
  let storageKey;
  if (mode === 'FRIDAY') storageKey = 'fridaySettings';
  else if (mode === 'ON CALL') storageKey = 'onCallSettings';
  else storageKey = 'slackSettings';
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(settings));
  } catch (error) {
    console.error(`Error saving ${mode} settings:`, error);
  }
};

// Update a specific setting for a mode
export const updateModeSetting = <K extends keyof ModeSpecificSettings>(
  mode: AppMode,
  key: K,
  value: ModeSpecificSettings[K]
): ModeSpecificSettings => {
  const currentSettings = getModeSettings(mode);
  const updatedSettings = { ...currentSettings, [key]: value };
  saveModeSettings(mode, updatedSettings);
  return updatedSettings;
};

// Get conversations for a specific mode
export const getModeConversations = (mode: AppMode) => {
  let storageKey;
  if (mode === 'FRIDAY') storageKey = 'conversationHistory_FRIDAY';
  else if (mode === 'ON CALL') storageKey = 'conversationHistory_ON_CALL';
  else storageKey = 'conversationHistory_SLACK';
  const stored = sessionStorage.getItem(storageKey);
  
  if (stored) {
    try {
      return JSON.parse(stored);
    } catch (error) {
      console.error(`Error parsing ${mode} conversations:`, error);
    }
  }
  
  return [];
};

// Save conversations for a specific mode
export const saveModeConversations = (mode: AppMode, conversations: any[]) => {
  let storageKey;
  if (mode === 'FRIDAY') storageKey = 'conversationHistory_FRIDAY';
  else if (mode === 'ON CALL') storageKey = 'conversationHistory_ON_CALL';
  else storageKey = 'conversationHistory_SLACK';
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(conversations));
  } catch (error) {
    console.error(`Error saving ${mode} conversations:`, error);
  }
};

// Clear conversations for a specific mode
export const clearModeConversations = (mode: AppMode) => {
  let storageKey;
  if (mode === 'FRIDAY') storageKey = 'conversationHistory_FRIDAY';
  else if (mode === 'ON CALL') storageKey = 'conversationHistory_ON_CALL';
  else storageKey = 'conversationHistory_SLACK';
  sessionStorage.removeItem(storageKey);
};

// Export data for a specific mode
export const exportModeData = (mode: AppMode) => {
  const conversations = getModeConversations(mode);
  const settings = getModeSettings(mode);
  
  // Get folders (shared between modes for now, but could be mode-specific if needed)
  const folders = sessionStorage.getItem('folders');
  const parsedFolders = folders ? JSON.parse(folders) : [];
  
  const data = {
    version: 4,
    mode: mode,
    history: conversations,
    folders: parsedFolders.filter((f: any) => f.type === 'chat'), // Only chat folders
    settings: settings,
    prompts: [], // Empty for now
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: 'application/json',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  
  const currentDate = () => {
    const date = new Date();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    return `${month}-${day}`;
  };
  
  link.download = `${mode.toLowerCase()}_mode_history_${currentDate()}.json`;
  link.href = url;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Import data for a specific mode
export const importModeData = (data: any, mode: AppMode) => {
  try {
    // Handle different data formats
    let importedConversations = [];
    let importedSettings = null;
    
    if (data.mode && data.mode === mode) {
      // This is a mode-specific export
      importedConversations = data.history || [];
      importedSettings = data.settings;
    } else if (Array.isArray(data)) {
      // Legacy format - just conversations
      importedConversations = data;
    } else if (data.history) {
      // Standard export format - filter by mode or assign to current mode
      importedConversations = data.history.map((conv: any) => ({
        ...conv,
        mode: conv.mode || mode // Assign to current mode if no mode specified
      })).filter((conv: any) => conv.mode === mode);
    }
    
    // Get existing conversations for this mode
    const existingConversations = getModeConversations(mode);
    
    // Merge conversations (avoid duplicates)
    const mergedConversations = [
      ...existingConversations,
      ...importedConversations
    ].filter(
      (conversation, index, self) =>
        index === self.findIndex((c) => c.id === conversation.id)
    );
    
    // Save merged conversations for this mode
    saveModeConversations(mode, mergedConversations);
    
    // If settings were included, save them
    if (importedSettings) {
      saveModeSettings(mode, importedSettings);
    }
    
    return {
      history: mergedConversations,
      settings: importedSettings,
      folders: data.folders || []
    };
  } catch (error) {
    console.error(`Error importing ${mode} data:`, error);
    throw error;
  }
}; 