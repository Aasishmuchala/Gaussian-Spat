'use client'

import { useCallback, useState, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { formatFileSize, generateId } from '@/lib/store'
import type { VideoFile } from '@/lib/types'
import { FileVideo, Upload, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'

interface DropZoneProps {
  onFilesAccepted: (files: VideoFile[]) => void
  maxFiles?: number
  maxSize?: number
  onUploadProgress?: (progress: Map<string, number>) => void
}

const ACCEPTED_TYPES = {
  'video/mp4': ['.mp4'],
  'video/quicktime': ['.mov'],
  'video/x-msvideo': ['.avi'],
  'video/webm': ['.webm'],
  'video/mpeg': ['.mpeg', '.mpg'],
}

export function DropZone({ onFilesAccepted, maxFiles = 10, maxSize = 10 * 1024 * 1024 * 1024, onUploadProgress }: DropZoneProps) {
  const [previews, setPreviews] = useState<Map<string, string>>(new Map())
  const [errors, setErrors] = useState<string[]>([])
  const [uploadProgress, setUploadProgress] = useState<Map<string, number>>(new Map())

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: { file: File; errors: { message: string }[] }[]) => {
    const errors: string[] = []

    rejectedFiles.forEach(({ file, errors: fileErrors }) => {
      fileErrors.forEach(err => {
        if (err.message.includes('too large')) {
          errors.push(`${file.name} exceeds ${formatFileSize(maxSize)}`)
        } else if (err.message.includes('type')) {
          errors.push(`${file.name} has unsupported type`)
        } else {
          errors.push(`${file.name}: ${err.message}`)
        }
      })
    })

    setErrors(errors)

    const videoFiles: VideoFile[] = acceptedFiles.map(file => {
      const id = generateId()
      const preview = URL.createObjectURL(file)
      setPreviews(prev => new Map(prev).set(id, preview))

      return {
        id,
        name: file.name,
        size: file.size,
        type: file.type,
        preview,
        status: 'pending' as const,
      }
    })

    if (videoFiles.length > 0) {
      onFilesAccepted(videoFiles)
    }
  }, [onFilesAccepted, maxSize])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles,
    maxSize,
    multiple: true,
  })

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer
          transition-all duration-200 ease-out
          ${isDragActive && !isDragReject ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30 scale-[1.02]' : ''}
          ${isDragReject ? 'border-red-500 bg-red-50 dark:bg-red-950/30' : ''}
          ${!isDragActive ? 'border-gray-300 dark:border-gray-700 hover:border-gray-400 dark:hover:border-gray-600' : ''}
        `}
      >
        <input {...getInputProps()} />

        <div className={`
          w-20 h-20 mx-auto mb-6 rounded-full flex items-center justify-center
          transition-colors duration-200
          ${isDragActive ? 'bg-blue-500' : 'bg-gray-100 dark:bg-gray-800'}
        `}>
          {isDragActive ? (
            <Upload className="w-10 h-10 text-white" />
          ) : (
            <FileVideo className="w-10 h-10 text-gray-400" />
          )}
        </div>

        <div className="space-y-2">
          <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
            {isDragActive ? 'Drop your videos here' : 'Drag & drop multi-view video files'}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            or <span className="text-blue-500 hover:text-blue-600">browse</span> to select files
          </p>
        </div>

        <div className="mt-6 flex flex-wrap justify-center gap-2">
          {['.mp4', '.mov', '.avi', '.webm'].map(ext => (
            <span key={ext} className="px-2 py-1 text-xs font-medium bg-gray-100 dark:bg-gray-800 rounded">
              {ext}
            </span>
          ))}
        </div>

        <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-950/30 rounded-lg border border-amber-200 dark:border-amber-800">
          <p className="text-xs text-amber-700 dark:text-amber-400">
            <span className="font-semibold">Multi-view required:</span> Upload 3-8 synchronized videos from different camera angles
          </p>
        </div>

        <div className="mt-4 flex justify-center gap-6 text-xs text-gray-400">
          <span>Max {maxFiles} files</span>
          <span>Max {formatFileSize(maxSize)} per file</span>
        </div>

        {isDragActive && (
          <div className="absolute inset-0 rounded-xl border-2 border-blue-500 animate-pulse-ring pointer-events-none" />
        )}
      </div>

      {errors.length > 0 && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-red-500" />
            <span className="text-sm font-medium text-red-700 dark:text-red-400">Errors</span>
          </div>
          <ul className="space-y-1">
            {errors.map((error, i) => (
              <li key={i} className="text-xs text-red-600 dark:text-red-400">{error}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export function VideoFileCard({
  file,
  onRemove,
  uploadProgress,
  onPreview
}: {
  file: VideoFile
  onRemove?: () => void
  uploadProgress?: number
  onPreview?: () => void
}) {
  const statusConfig = {
    pending: { icon: <div className="w-3 h-3 rounded-full bg-gray-400" />, label: 'Pending', color: 'text-gray-500' },
    uploading: { icon: <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />, label: 'Uploading', color: 'text-blue-500' },
    uploaded: { icon: <CheckCircle className="w-4 h-4 text-green-500" />, label: 'Uploaded', color: 'text-green-500' },
    processing: { icon: <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />, label: 'Processing', color: 'text-yellow-500' },
    complete: { icon: <CheckCircle className="w-4 h-4 text-green-600" />, label: 'Complete', color: 'text-green-600' },
    error: { icon: <AlertCircle className="w-4 h-4 text-red-500" />, label: 'Error', color: 'text-red-500' },
  }

  const status = statusConfig[file.status]

  return (
    <div className="group relative bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden transition-all duration-200 hover:shadow-md">
      {file.preview && (
        <div className="aspect-video bg-gray-100 dark:bg-gray-900 cursor-pointer" onClick={onPreview}>
          <video src={file.preview} className="w-full h-full object-cover" muted playsInline />
        </div>
      )}

      <div className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium truncate" title={file.name}>{file.name}</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">{formatFileSize(file.size)}</p>
          </div>
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1 ${status.color}`}>
              {status.icon}
              <span className="text-xs">{status.label}</span>
            </div>
            {onRemove && (
              <button onClick={onRemove} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors opacity-0 group-hover:opacity-100">
                <X className="w-4 h-4 text-gray-400 hover:text-red-500" />
              </button>
            )}
          </div>
        </div>

        {/* Upload progress bar */}
        {uploadProgress !== undefined && file.status === 'uploading' && (
          <div className="mt-2">
            <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">{Math.round(uploadProgress)}%</p>
          </div>
        )}
      </div>
    </div>
  )
}