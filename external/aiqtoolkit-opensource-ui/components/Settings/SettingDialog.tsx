import { FC, useContext, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'next-i18next';
import HomeContext from '@/pages/api/home/home.context';
import toast from 'react-hot-toast';

interface Props {
  open: boolean;
  onClose: () => void;
}

export const SettingDialog: FC<Props> = ({ open, onClose }) => {
  const { t } = useTranslation('settings');
  const modalRef = useRef<HTMLDivElement>(null);
  const {
    state: { lightMode, chatCompletionURL, webSocketURL, webSocketSchema: schema, expandIntermediateSteps, intermediateStepOverride, enableIntermediateSteps, webSocketSchemas, currentMode },
    dispatch: homeDispatch,
  } = useContext(HomeContext);

  const [theme, setTheme] = useState(lightMode);
  const [chatCompletionEndPoint, setChatCompletionEndPoint] = useState(chatCompletionURL || '');
  const [webSocketEndPoint, setWebSocketEndPoint] = useState(webSocketURL || '');
  const [webSocketSchema, setWebSocketSchema] = useState(schema || '');
  const [isIntermediateStepsEnabled, setIsIntermediateStepsEnabled] = useState(enableIntermediateSteps);
  const [detailsToggle, setDetailsToggle] = useState(expandIntermediateSteps);
  const [intermediateStepOverrideToggle, setIntermediateStepOverrideToggle] = useState(intermediateStepOverride);

  // Load mode-specific settings when dialog opens
  useEffect(() => {
    if (open) {
      import('@/utils/app/modeSettings').then(({ getModeSettings }) => {
        const modeSettings = getModeSettings(currentMode);
        setChatCompletionEndPoint(modeSettings.chatCompletionURL);
        setWebSocketEndPoint(modeSettings.webSocketURL);
        setWebSocketSchema(modeSettings.webSocketSchema);
        setIsIntermediateStepsEnabled(modeSettings.enableIntermediateSteps);
        setDetailsToggle(modeSettings.expandIntermediateSteps);
        setIntermediateStepOverrideToggle(modeSettings.intermediateStepOverride);
      });
    }
  }, [open, currentMode]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    if (open) {
      window.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      window.removeEventListener('mousedown', handleClickOutside);
    };
  }, [open, onClose]);

  const handleSave = () => {
    if(!chatCompletionEndPoint || !webSocketEndPoint) {
      toast.error('Please fill all the fields to save settings');
      return;
    }

    // Save theme globally (shared between modes)
    homeDispatch({ field: 'lightMode', value: theme });

    // Save mode-specific settings
    import('@/utils/app/modeSettings').then(({ saveModeSettings }) => {
      const modeSettings = {
        chatHistory: false, // Will be preserved from existing settings
        chatCompletionURL: chatCompletionEndPoint,
        webSocketMode: false, // Will be preserved from existing settings
        webSocketConnected: false, // Will be preserved from existing settings
        webSocketURL: webSocketEndPoint,
        webSocketSchema: webSocketSchema,
        enableIntermediateSteps: isIntermediateStepsEnabled || false,
        expandIntermediateSteps: detailsToggle || false,
        intermediateStepOverride: intermediateStepOverrideToggle || false,
      };

      // Get existing settings and merge to preserve toggle states
      import('@/utils/app/modeSettings').then(({ getModeSettings }) => {
        const existingSettings = getModeSettings(currentMode);
        const mergedSettings = {
          ...existingSettings,
          ...modeSettings,
        };
        
        saveModeSettings(currentMode, mergedSettings);

        // Update legacy fields for immediate UI update
        homeDispatch({ field: 'chatCompletionURL', value: chatCompletionEndPoint });
        homeDispatch({ field: 'webSocketURL', value: webSocketEndPoint });
        homeDispatch({ field: 'webSocketSchema', value: webSocketSchema });
        homeDispatch({ field: 'expandIntermediateSteps', value: detailsToggle });
        homeDispatch({ field: 'intermediateStepOverride', value: intermediateStepOverrideToggle });
        homeDispatch({ field: 'enableIntermediateSteps', value: isIntermediateStepsEnabled });

        toast.success(`Settings saved for ${currentMode} mode`);
        onClose();
      });
    });
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm z-50 dark:bg-opacity-20">
      <div
        ref={modalRef}
        className="w-full max-w-md bg-white dark:bg-[#202123] rounded-2xl shadow-lg p-6 transform transition-all relative"
      >
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          {t('Settings')} - {currentMode} Mode
        </h2>

        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">{t('Theme')}</label>
        <select
          className="w-full mt-1 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none"
          value={theme}
          onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
        >
          <option value="dark">{t('Dark mode')}</option>
          <option value="light">{t('Light mode')}</option>
        </select>
      
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">{t('HTTP URL for Chat Completion')}</label>
        <input
          type="text"
          value={chatCompletionEndPoint}
          onChange={(e) => setChatCompletionEndPoint(e.target.value)}
          className="w-full mt-1 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none"
        />

        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">{t('WebSocket URL for Chat Completion')}</label>
        <input
          type="text"
          value={webSocketEndPoint}
          onChange={(e) => setWebSocketEndPoint(e.target.value)}
          className="w-full mt-1 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none"
        />

        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">{t('WebSocket Schema')}</label>
        <select
          className="w-full mt-1 p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none"
          value={webSocketSchema}
          onChange={(e) => {
            setWebSocketSchema(e.target.value)}
          }
        >
          {webSocketSchemas?.map((schema) => (
            <option key={schema} value={schema}>
              {schema}
            </option>
          ))}
        </select>

        <div className="flex align-middle text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">
          <input
            type="checkbox"
            id="enableIntermediateSteps"
            checked={isIntermediateStepsEnabled}
            onChange={ () => {
              setIsIntermediateStepsEnabled(!isIntermediateStepsEnabled)
            }}
            className="mr-2"
          />
          <label
            htmlFor="enableIntermediateSteps"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Enable Intermediate Steps
          </label>
        </div>

        <div className="flex align-middle text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">
          <input
            type="checkbox"
            id="detailsToggle"
            checked={detailsToggle}
            onChange={ () => {
              setDetailsToggle(!detailsToggle)
            }}
            disabled={!isIntermediateStepsEnabled}
            className="mr-2"
          />
          <label
            htmlFor="detailsToggle"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Expand Intermediate Steps by default
          </label>
        </div>

        <div className="flex align-middle text-sm font-medium text-gray-700 dark:text-gray-300 mt-4">
          <input
            type="checkbox"
            id="intermediateStepOverrideToggle"
            checked={intermediateStepOverrideToggle}
            onChange={ () => {
              setIntermediateStepOverrideToggle(!intermediateStepOverrideToggle)
            }}
            disabled={!isIntermediateStepsEnabled}
            className="mr-2"
          />
          <label
            htmlFor="intermediateStepOverrideToggle"
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            Override intermediate Steps with same Id
          </label>
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <button
            className="px-4 py-2 bg-gray-300 dark:bg-gray-600 text-gray-900 dark:text-white rounded-md hover:bg-gray-400 dark:hover:bg-gray-500 focus:outline-none"
            onClick={onClose}
          >
            {t('Cancel')}
          </button>
          <button
            className="px-4 py-2 bg-[#76b900] text-white rounded-md hover:bg-[#5a9100] focus:outline-none"
            onClick={handleSave}
          >
            {t('Save')}
          </button>
        </div>
      </div>
    </div>
  );
};
