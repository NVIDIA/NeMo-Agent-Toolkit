import React from 'react';
import { getInitials } from '@/utils/app/helper';

export const UserAvatar = ({src = '', height = 30, width = 30}) =>  {
    const profilePicUrl = src || ``

    const onError = (event: { target: { src: string; }; }) => {
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <rect width="100%" height="100%" fill="#fff"/>
            <text x="50%" y="50%" alignment-baseline="middle" text-anchor="middle" fill="#333" font-size="16" font-family="Arial, sans-serif">
                user
            </text>
        </svg>`;
        event.target.src = `data:image/svg+xml;base64,${window.btoa(svg)}`;
    };

    return <img 
        src={profilePicUrl} 
        alt={'user-avatar'}
        width={width}
        height={height}
        title={'user-avatar'}
        className='rounded-full max-w-full h-auto border border-[#76b900]'
        onError={onError}
    />

}