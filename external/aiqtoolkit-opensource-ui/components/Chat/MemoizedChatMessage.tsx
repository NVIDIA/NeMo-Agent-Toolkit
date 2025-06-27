import { FC, memo } from "react";
import isEqual from "lodash/isEqual";
import { ChatMessage, Props } from "./ChatMessage";

export const MemoizedChatMessage: FC<Props> = memo(
    ChatMessage,
    (prevProps, nextProps) => {
        // componenent will render if new props are only different than previous props (to prevent unnecessary re-rendering)
        const shouldRender = isEqual(prevProps.message, nextProps.message);
        return shouldRender;
    }
);
