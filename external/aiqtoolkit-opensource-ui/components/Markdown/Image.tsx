import React, { memo, useMemo, useRef, useState, useCallback } from "react";
import Loading from "./Loading";
import { IconExclamationCircle, IconMaximize, IconX } from "@tabler/icons-react";

export const Image = memo(({ src, alt, ...props }) => {
  const imgRef = useRef(null);
  const [error, setError] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleImageError = () => {
    console.error(`Image failed to load: ${src}`);
    setError(true);
  };

  const toggleFullscreen = useCallback(() => {
    setIsFullscreen((prev) => !prev);
  }, []);

  const imageElement = useMemo(() => {
    if (src === "loading") {
      return <Loading message="Loading..." type="image" />;
    }

    return (
      <>
        {/* Image Container */}
        <div className="relative">
          {error ? (
            <div className="flex items-center justify-center p-4 bg-slate-50 rounded-lg border border-slate-200">
              <IconExclamationCircle className="w-5 h-5 text-red-500 mr-2" />
              <p className="text-red-600 text-sm">
                Failed to load image with src: {src.slice(0, 50) + (src.length > 50 ? "..." : "")}
              </p>
            </div>
          ) : (
            <div className="relative">
              {/* Image */}
              <img
                ref={imgRef}
                src={src}
                alt={alt || "image"}
                onError={handleImageError}
                className="object-cover rounded-lg border border-slate-100 shadow-xs cursor-pointer"
                onClick={toggleFullscreen}
                {...props}
              />
              {/* Fullscreen Mode */}
              {isFullscreen && !error && (
                <div
                  className="fixed inset-0 bg-black/95 flex items-center justify-center z-50"
                  onClick={toggleFullscreen}
                >
                  <div className="relative">
                    <img
                      src={src}
                      alt={alt || "image"}
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>        
      </>
    );
  }, [src, alt, error, isFullscreen, toggleFullscreen]);

  return imageElement;
}, (prevProps, nextProps) => prevProps.src === nextProps.src);
