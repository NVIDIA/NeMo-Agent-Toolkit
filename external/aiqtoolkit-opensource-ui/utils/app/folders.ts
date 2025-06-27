import { FolderInterface } from '@/types/folder';

export const saveFolders = (folders: FolderInterface[]) => {
  sessionStorage.setItem('folders', JSON.stringify(folders));
};
