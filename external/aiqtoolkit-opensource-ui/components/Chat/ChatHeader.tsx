'use client'

import React, { useContext, useState, useRef, useEffect } from 'react';
import { 
  IconArrowsSort, 
  IconMobiledataOff, 
  IconSun, 
  IconMoonFilled, 
  IconUserFilled,
  IconChevronLeft,
  IconChevronRight
} from '@tabler/icons-react';
import HomeContext from '@/pages/api/home/home.context';

interface ChatHeaderProps {
    webSocketModeRef?: React.MutableRefObject<boolean>;
}

export const ChatHeader = ({ webSocketModeRef }: ChatHeaderProps) => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isExpanded, setIsExpanded] = useState(process?.env?.NEXT_PUBLIC_RIGHT_MENU_OPEN === 'true' ? true : false);
    const menuRef = useRef<HTMLDivElement>(null);

    const {
        state: {
          chatHistory,
          webSocketMode,
          webSocketConnected,
          lightMode,
          selectedConversation,
          currentMode
        },
        dispatch: homeDispatch,
      } = useContext(HomeContext);
    
    
    const handleLogin = () => {
        console.log('Login clicked');
        setIsMenuOpen(false);
    };

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
                setIsMenuOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className={`top-0 z-10 flex justify-center items-center h-12 ${(selectedConversation?.messages?.length || 0) === 0 ? 'bg-none' : 'bg-[#76b900] sticky'}  py-2 px-4 text-sm text-white dark:border-none dark:bg-black dark:text-neutral-200`}>
            {
                (selectedConversation?.messages?.length || 0) > 0 ? 
                <div className={`absolute top-6 left-1/2 transform -translate-x-1/2 -translate-y-1/2`}>
                    <span className="text-lg font-semibold text-white">{currentMode}</span>
                </div> 
                :
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 mx-auto flex flex-col space-y-5 md:space-y-10 px-3 pt-5 md:pt-12 sm:max-w-[600px] text-center">
                  <div className="text-3xl font-semibold text-gray-800 dark:text-white">
                    Hi, I'm {currentMode}
                  </div>
                  <div className="text-lg text-gray-600 dark:text-gray-400">
                    How can I assist you today?
                  </div>
                </div>
            } 

            {/* Collapsible Menu */}
            <div className={`fixed right-0 top-0 h-12 flex items-center transition-all duration-300 ${isExpanded ? 'mr-2' : 'mr-2'}`}>
                <button 
                    onClick={() => {
                        setIsExpanded(!isExpanded)}
                    }
                    className="flex p-1 text-black dark:text-white transition-colors"
                >
                    {isExpanded ? <IconChevronRight size={20} /> : <IconChevronLeft size={20} />}
                </button>

                <div className={`flex sm: gap-1 md:gap-4 overflow-hidden transition-all duration-300 ${isExpanded ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>
                    {/* Chat History Toggle */}
                    <div className="flex items-center gap-2 whitespace-nowrap">
                        <label className="flex items-center gap-2 cursor-pointer flex-shrink-0">
                            <span className="text-sm font-medium text-black dark:text-white">Chat History</span>
                            <div
                                onClick={() => {
                                    // Update mode-specific setting
                                    import('@/utils/app/modeSettings').then(({ updateModeSetting }) => {
                                        const newValue = !chatHistory;
                                        updateModeSetting(currentMode, 'chatHistory', newValue);
                                        homeDispatch({
                                            field: 'chatHistory',
                                            value: newValue,
                                        });
                                    });
                                }}
                                className={`relative inline-flex h-5 w-10 items-center cursor-pointer rounded-full transition-colors duration-300 ease-in-out ${
                                    chatHistory ? 'bg-black dark:bg-[#76b900]' : 'bg-gray-200'
                                }`}
                            >
                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-300 ease-in-out ${
                                    chatHistory ? 'translate-x-6' : 'translate-x-0'
                                }`} />
                            </div>
                        </label>
                    </div>

                    {/* WebSocket Mode Toggle */}
                    <div className="flex items-center gap-2 whitespace-nowrap">
                        <label className="flex items-center gap-2 cursor-pointer flex-shrink-0">
                            <span className={`flex items-center gap-1 justify-evenly text-sm font-medium text-black dark:text-white`}>
                                WebSocket{' '}
                                {webSocketMode && (
                                    webSocketConnected ? <IconArrowsSort size={18} color="black" /> : <IconMobiledataOff size={18} color="black" />
                                )}
                            </span>
                            <div
                                onClick={() => {
                                    // Update mode-specific setting
                                    import('@/utils/app/modeSettings').then(({ updateModeSetting }) => {
                                        const newValue = !webSocketMode;
                                        updateModeSetting(currentMode, 'webSocketMode', newValue);
                                        if (webSocketModeRef) {
                                            webSocketModeRef.current = newValue;
                                        }
                                        homeDispatch({
                                            field: 'webSocketMode',
                                            value: newValue,
                                        });
                                    });
                                }}
                                className={`relative inline-flex h-5 w-10 items-center cursor-pointer rounded-full transition-colors duration-300 ease-in-out ${
                                    webSocketMode ? 'bg-black dark:bg-[#76b900]' : 'bg-gray-200'
                                }`}
                            >
                                <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform duration-300 ease-in-out ${
                                    webSocketMode ? 'translate-x-6' : 'translate-x-0'
                                }`} />
                            </div>
                        </label>
                    </div>

                    {/* Theme Toggle Button */}
                    <div className="flex items-center dark:text-white text-black transition-colors duration-300">
                        <button
                            onClick={() => {
                                const newMode = lightMode === 'dark' ? 'light' : 'dark';
                                homeDispatch({
                                    field: 'lightMode',
                                    value: newMode,
                                });
                            }}
                            className="rounded-full flex items-center justify-center bg-none dark:bg-gray-700 transition-colors duration-300 focus:outline-none"
                        >
                            {lightMode === 'dark' ? (
                                <IconSun className="w-6 h-6 text-yellow-500 transition-transform duration-300" />
                            ) : (
                                <IconMoonFilled className="w-6 h-6 text-gray-800 transition-transform duration-300" />
                            )}
                        </button>
                    </div>

                    {/* User Icon with Dropdown Menu */}
                    <div className="relative" ref={menuRef}>
                        <button 
                            onClick={() => setIsMenuOpen(!isMenuOpen)}
                            className="flex items-center dark:text-white text-black cursor-pointer"
                        >
                            <IconUserFilled size={20} />
                        </button>
                        {isMenuOpen && (
                            <div className="absolute right-0 mt-2 px-2 w-auto rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5">
                                <div className="py-1">
                                    <button
                                        onClick={handleLogin}
                                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700"
                                    >
                                        Login
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};