import { IconUserPentagon } from '@tabler/icons-react';
import React from 'react';
export const AgentAvatar = ({height = 7, width = 7}) =>  {
    return (
        <div className={`w-${width} h-${height} flex justify-center items-center rounded-full bg-[#004D3C] text-white`} title="Agent">
                <IconUserPentagon size={25}/>
        </div>
    )
}