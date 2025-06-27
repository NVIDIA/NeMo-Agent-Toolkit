import { IconCheck, IconClipboard, IconDownload } from '@tabler/icons-react';
import { FC, memo, MouseEvent, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

import { useTranslation } from 'next-i18next';

import {
  generateRandomString,
  programmingLanguages,
} from '@/utils/app/codeblock';

interface Props {
  language: string;
  value: string;
}

export const CodeBlock: FC<Props> = memo(({ language, value }) => {
  const { t } = useTranslation('markdown');
  const [isCopied, setIsCopied] = useState<boolean>(false);

  // Ensure value is a valid JSON string
  if (language === 'json') {
    try {
      value = value.replaceAll("'", '"');
    } catch (error) {
      console.log(error);
    }
  }

  const formattedValue = (() => {
          try {
            return JSON.stringify(JSON.parse(value), null, 2);
          } catch {
            return value; // Return the original value if parsing fails
          }
        })()


  const copyToClipboard = (e) => {
    e?.preventDefault();
    e?.stopPropagation();
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      return;
    }

    navigator.clipboard.writeText(formattedValue).then(() => {
      setIsCopied(true);

      setTimeout(() => {
        setIsCopied(false);
      }, 2000);
    });
  };

  const downloadAsFile = (e) => {
    e?.preventDefault();
    e?.stopPropagation();
    const fileExtension = programmingLanguages[language] || '.file';
    const suggestedFileName = `file-${generateRandomString(
      3,
      true,
    )}${fileExtension}`;
    // const fileName = window.prompt(
    //   t('Enter file name') || '',
    //   suggestedFileName,
    // );

    if (!suggestedFileName) {
      return; // User pressed cancel on prompt
    }

    const blob = new Blob([formattedValue], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = suggestedFileName;
    link.href = url;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="codeblock relative font-sans text-[16px]">
      <div className="flex items-center justify-between py-1.5 px-4">
        <span className="text-xs lowercase text-white">{language}</span>

        <div className="flex items-center">
          <button
            className="flex gap-1.5 items-center rounded bg-none p-1 text-xs text-white"
            onClick={(e) => copyToClipboard(e)}
          >
            {isCopied ? <IconCheck size={18} /> : <IconClipboard size={18} />}
            {isCopied ? t('Copied!') : t('Copy code')}
          </button>
          <button
            className="flex items-center rounded bg-none p-1 text-xs text-white"
            onClick={(e) => downloadAsFile(e)}
          >
            <IconDownload size={18} />
          </button>
        </div>
      </div>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{
          margin: 0,
          // width: 'max-content',
          maxWidth: '200ch',
          maxHeight: '50vh',
          display: 'block',
          boxSizing: 'border-box',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          overflowX: 'auto',
          overflowY: 'auto', 
        }}
        wrapLongLines={true} // Ensures long lines wrap instead of forcing width expansion
      >
        {formattedValue}
      </SyntaxHighlighter>
    </div>
  );
});
CodeBlock.displayName = 'CodeBlock';
