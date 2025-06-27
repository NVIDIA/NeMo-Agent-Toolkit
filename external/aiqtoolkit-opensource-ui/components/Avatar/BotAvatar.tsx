import React from 'react';

export const BotAvatar = ({height = 30, width = 30, src= ''}) =>  {

    const onError = (event: { target: { src: string; }; }) => {
        console.error('error loading bot avatar');
        event.target.src = `nvidia.jpg`;
    };

    return <img 
        src={src} 
        alt="bot-avatar" 
        width={width}
        height={height}
        className='rounded-full max-w-full h-auto'
        onError={onError}
    />
}