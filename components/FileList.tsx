'use client'

import { VideoFileCard } from './DropZone'
import type { VideoFile } from '@/lib/types'
import { Film, Trash2 } from 'lucide-react'

interface FileListProps {
  files: VideoFile[]
  onRemove?: (fileId: string) => void
  onPreview?: (fileId: string) => void
}

export function FileList({ files, onRemove, onPreview }: FileListProps) {
  if (files.length === 0) {
    return null
  }

  return (
    <div className="w-full mt-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Film className="w-5 h-5 text-gray-500" />
          <h3 className="text-lg font-medium">
            Uploaded Videos ({files.length})
          </h3>
        </div>

        {onRemove && files.length > 0 && (
          <button
            onClick={() => files.forEach(f => onRemove?.(f.id))}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/30 rounded-lg transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        )}
      </div>

      {/* File grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {files.map(file => (
          <VideoFileCard
            key={file.id}
            file={file}
            onRemove={onRemove ? () => onRemove(file.id) : undefined}
            onPreview={onPreview ? () => onPreview(file.id) : undefined}
          />
        ))}
      </div>

      {/* Summary */}
      <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500 dark:text-gray-400">
            Total files: {files.length}
          </span>
          <span className="text-gray-500 dark:text-gray-400">
            {files.filter(f => f.status === 'uploaded' || f.status === 'complete').length} ready for processing
          </span>
        </div>
      </div>
    </div>
  )
}