import React, { useRef } from 'react';
import { Upload, FileText, Plus, FileUp } from 'lucide-react';

export default function UploadDropzone({
  id,
  title,
  subtitle,
  icon: Icon = Upload,
  buttonLabel = 'Browse Files',
  multiple = false,
  onFilesSelected,
  acceptedTypes = '.pdf,application/pdf'
}) {
  const fileInputRef = useRef(null);

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const filesArray = Array.from(e.target.files);
      onFilesSelected(filesArray);
      // Reset input value so re-selecting same file triggers change
      e.target.value = '';
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const filesArray = Array.from(e.dataTransfer.files).filter(
        file => file.type === 'application/pdf' || file.name.endsWith('.pdf')
      );
      if (filesArray.length > 0) {
        onFilesSelected(filesArray);
      }
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className="border-2 border-dashed border-[#1F2937] hover:border-purple-500/50 rounded-2xl p-6 sm:p-8 text-center bg-[#0B1020] hover:bg-[#0B1020]/90 transition-all group flex flex-col items-center justify-center space-y-3"
    >
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        id={id}
        type="file"
        accept={acceptedTypes}
        multiple={multiple}
        onChange={handleFileChange}
        className="hidden"
      />

      <div className="w-12 h-12 rounded-2xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center group-hover:scale-110 transition-transform">
        <Icon className="w-6 h-6" />
      </div>

      <div>
        <h4 className="font-heading font-bold text-sm text-white group-hover:text-purple-300 transition-colors">
          {title}
        </h4>
        <p className="text-xs text-slate-400 mt-0.5">
          {subtitle}
        </p>
      </div>

      {/* Prominent Primary Browse Button */}
      <button
        type="button"
        onClick={handleButtonClick}
        className="px-5 py-2.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs shadow-lg shadow-purple-500/25 flex items-center space-x-2 transition-all cursor-pointer"
      >
        <FileUp className="w-4 h-4" />
        <span>{buttonLabel}</span>
      </button>

      {/* Supporting Text */}
      <span className="text-[11px] font-medium text-slate-500">
        or drag & drop your PDF file{multiple ? 's' : ''} here
      </span>
    </div>
  );
}
