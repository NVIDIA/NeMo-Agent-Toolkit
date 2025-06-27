import { FC, useEffect, useState } from 'react';
import { BotAvatar } from '@/components/Avatar/BotAvatar';

interface Props {
  statusUpdateText: string;
}

export const ChatLoader: FC<Props> = ({ statusUpdateText = '' }) => {
  const config = {
    initialDelay: 500,
    delayMultiplier: 6000,
    statusMessages: [statusUpdateText],
  };

  const [currentMessage, setCurrentMessage] = useState(''); // Initialize with empty string

  useEffect(() => {
    const timers = config.statusMessages.map((message, index) => {
      const delay = index === 0 ? config.initialDelay : config.initialDelay + (index * config.delayMultiplier);
      return setTimeout(() => {
        setCurrentMessage(message);
      }, delay);
    });

    return () => {
      timers.forEach((timer) => clearTimeout(timer));
    };
  }, []);

  return (
    <div
      className="group border-b border-black/10 bg-gray-50 text-gray-800 dark:border-gray-900/50 dark:bg-[#444654] dark:text-gray-100"
      style={{ overflowWrap: 'anywhere' }}
    >
      <div className="relative m-auto flex p-4 text-base sm:w-[95%] md:w-[92%] lg:w-[93%] 2xl:w-[59%] md:gap-6 md:py-6 lg:px-0">
        <div className="min-w-[40px] items-end">
          <BotAvatar src={'nvidia.jpg'} size={30} />
        </div>
        <div className="flex items-center">
          {/* Status Update Text with Green Blinking Caret */}
          <span className="cursor-default">{currentMessage}<span className="text-[#76b900] animate-blink">▍</span></span>
        </div>
      </div>
    </div>
  );
};