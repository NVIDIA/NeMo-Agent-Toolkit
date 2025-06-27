// First, define the Video component at module level
'use client'

import { memo, useMemo, useRef } from "react";
import Loading from "@/components/Markdown/Loading";

export const Video = memo(({ 
    src, 
    controls = true, 
    muted = false, 
    ...props 
  }) => {
    // Use ref to maintain stable reference for video element
    const videoRef = useRef(null);
  
    // Memoize the video element to prevent re-renders from context changes
    const videoElement = useMemo(() => {
      if (src === 'loading') {
        return <Loading message='Loading...' type='image' />;
      }
  
      return (
        <video
          ref={videoRef}
          src={src}
          controls={controls}
          autoPlay={false}
          loop={false}
          muted={muted}
          playsInline={false}
          className="rounded-md border border-slate-400 shadow-sm object-cover"
          {...props}
        >
          Your browser does not support the video tag.
        </video>
      );
    }, [src, controls, muted]); // Only dependencies that should cause a re-render
  
    return videoElement;
  }, (prevProps, nextProps) => {
    return prevProps.src === nextProps.src;
  });