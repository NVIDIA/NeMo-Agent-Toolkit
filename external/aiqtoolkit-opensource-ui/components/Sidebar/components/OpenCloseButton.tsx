import { IconMenu2 } from '@tabler/icons-react';

interface Props {
  onClick: any;
  side: 'left' | 'right';
}

export const CloseSidebarButton = ({ onClick, side }: Props) => {
  return (
    <button
      className={`mt-1 fixed top-5 ${
        side === 'right' ? 'right-[270px]' : 'left-[270px]'
      } z-50 h-7 w-7 hover:text-gray-400 dark:text-white dark:hover:text-gray-300 sm:top-0.5 sm:${
        side === 'right' ? 'right-[270px]' : 'left-[270px]'
      } sm:h-8 sm:w-8 sm:text-neutral-700`}
      onClick={onClick}
    >
      <IconMenu2 className="text-black dark:text-white" size={18} />
    </button>
  );
};

export const OpenSidebarButton = ({ onClick, side }: Props) => {
  return (
    <button
      className={`mt-1 fixed top-2.5 ${
        side === 'right' ? 'right-2' : 'left-2'
      } z-50 h-7 w-7 hover:text-gray-400 dark:text-white dark:hover:text-gray-300 sm:top-0.5 sm:${
        side === 'right' ? 'right-2' : 'left-2'
      } sm:h-8 sm:w-8 sm:text-neutral-700`}
      onClick={onClick}
    >
      <IconMenu2 className="text-black dark:text-white" size={18} />
    </button>
  );
};