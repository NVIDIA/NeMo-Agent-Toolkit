import { useContext } from 'react';
import { AppMode } from '@/pages/api/home/home.state';
import HomeContext from '@/pages/api/home/home.context';

export const ModeSwitcher = () => {
  const {
    state: { currentMode },
    dispatch,
  } = useContext(HomeContext);

  const handleModeSwitch = (mode: AppMode) => {
    console.log(`🔄 Switching to ${mode} mode`);
    
    // Update the current mode
    dispatch({ field: 'currentMode', value: mode });
    sessionStorage.setItem('currentMode', mode);
    
    // Dynamically update the backend URLs based on mode
    import('@/utils/app/modeSettings').then(({ getModeSettings, updateModeSetting }) => {
      const modeSettings = getModeSettings(mode);
      
      console.log(`📡 ${mode} connecting to: ${modeSettings.webSocketURL}`);
      
      // Update the global settings to point to the correct backend
      dispatch({ field: 'chatCompletionURL', value: modeSettings.chatCompletionURL });
      dispatch({ field: 'webSocketURL', value: modeSettings.webSocketURL });
      dispatch({ field: 'webSocketMode', value: modeSettings.webSocketMode });
      dispatch({ field: 'chatHistory', value: modeSettings.chatHistory });
      dispatch({ field: 'enableIntermediateSteps', value: modeSettings.enableIntermediateSteps });
      dispatch({ field: 'expandIntermediateSteps', value: modeSettings.expandIntermediateSteps });
      dispatch({ field: 'intermediateStepOverride', value: modeSettings.intermediateStepOverride });
      
      // Update sessionStorage with the new URLs for immediate use
      sessionStorage.setItem('chatCompletionURL', modeSettings.chatCompletionURL);
      sessionStorage.setItem('webSocketURL', modeSettings.webSocketURL);
      sessionStorage.setItem('webSocketMode', modeSettings.webSocketMode.toString());
      sessionStorage.setItem('chatHistory', modeSettings.chatHistory.toString());
      sessionStorage.setItem('enableIntermediateSteps', modeSettings.enableIntermediateSteps.toString());
      
      // Update mode-specific settings
      if (mode === 'FRIDAY') {
        dispatch({ field: 'fridaySettings', value: modeSettings });
      } else if (mode === 'ON CALL') {
        dispatch({ field: 'onCallSettings', value: modeSettings });
      } else {
        dispatch({ field: 'slackSettings', value: modeSettings });
      }
    });
  };

  return (
    <div className="flex flex-col space-y-2 p-2 border-b border-white/20 mb-2">
      <div className="text-xs text-white/60 font-medium uppercase tracking-wider">
        Mode
      </div>
      <div className="flex space-x-1">
        <button
          onClick={() => handleModeSwitch('FRIDAY')}
          className={`flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors duration-200 ${
            currentMode === 'FRIDAY'
              ? 'bg-blue-600 text-white border border-blue-500'
              : 'bg-gray-700/50 text-white/70 border border-white/20 hover:bg-gray-600/50 hover:text-white'
          }`}
        >
          FRIDAY
        </button>
        <button
          onClick={() => handleModeSwitch('ON CALL')}
          className={`flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors duration-200 ${
            currentMode === 'ON CALL'
              ? 'bg-green-600 text-white border border-green-500'
              : 'bg-gray-700/50 text-white/70 border border-white/20 hover:bg-gray-600/50 hover:text-white'
          }`}
        >
          ON CALL
        </button>
        <button
          onClick={() => handleModeSwitch('SLACK')}
          className={`flex-1 px-3 py-2 text-xs font-medium rounded-md transition-colors duration-200 ${
            currentMode === 'SLACK'
              ? 'bg-purple-600 text-white border border-purple-500'
              : 'bg-gray-700/50 text-white/70 border border-white/20 hover:bg-gray-600/50 hover:text-white'
          }`}
        >
          SLACK
        </button>
      </div>
    </div>
  );
}; 