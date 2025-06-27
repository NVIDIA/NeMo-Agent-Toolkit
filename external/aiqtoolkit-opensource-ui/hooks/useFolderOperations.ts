import { saveFolders } from '@/utils/app/folders'; // Adjust according to your utility functions' locations
import { v4 as uuidv4 } from 'uuid';

export const useFolderOperations = ({folders, dispatch}) => {

  const handleCreateFolder = (name, type) => {
    const newFolder = {
      id: uuidv4(), // Ensure you have uuid imported or an alternative way to generate unique ids
      name,
      type,
    };

    const updatedFolders = [...folders, newFolder];
    dispatch({ field: 'folders', value: updatedFolders });
    saveFolders(updatedFolders); // Assuming you have a utility function to persist folders change
  };

  const handleDeleteFolder = (folderId) => {
    const updatedFolders = folders.filter(folder => folder.id !== folderId);
    dispatch({ field: 'folders', value: updatedFolders });
    saveFolders(updatedFolders); // Persist the updated list after deletion
  };

  const handleUpdateFolder = (folderId, name) => {
    const updatedFolders = folders.map(folder => 
      folder.id === folderId ? { ...folder, name } : folder
    );
    dispatch({ field: 'folders', value: updatedFolders });
    saveFolders(updatedFolders); // Persist the updated list
  };

  return { handleCreateFolder, handleDeleteFolder, handleUpdateFolder };
};

