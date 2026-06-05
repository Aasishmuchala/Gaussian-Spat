'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import { DropZone, VideoFileCard } from '@/components/DropZone'
import { FileList } from '@/components/FileList'
import { usePipelineStore, generateId, formatFileSize } from '@/lib/store'
import type { VideoFile, PipelineStatus } from '@/lib/types'
import { Play, Loader2, CheckCircle, AlertCircle, Upload, Scissors, Camera, Cpu, Eye, X } from 'lucide-react'

// Pipeline stages with weights (must sum to 100)
const PIPELINE_STAGES: { stage: PipelineStatus['stage']; weight: number; icon: typeof Upload; label: string; description: string }[] = [
  { stage: 'uploading', weight: 10, icon: Upload, label: 'Uploading', description: 'Sending videos to server' },
  { stage: 'extracting_frames', weight: 15, icon: Scissors, label: 'Extracting Frames', description: 'FFmpeg extracting frames from videos' },
  { stage: 'running_colmap', weight: 30, icon: Camera, label: 'COLMAP SfM', description: '3D reconstruction with Structure-from-Motion' },
  { stage: 'training_4dgs', weight: 40, icon: Cpu, label: 'Training 4DGS', description: 'Training Gaussian splatting model' },
  { stage: 'starting_viewer', weight: 5, icon: Eye, label: 'Starting Viewer', description: 'Launching interactive 3D viewer' },
]

function calculateOverallProgress(currentStage: PipelineStatus['stage'], stageProgress: number): number {
  let completedWeight = 0
  for (const stage of PIPELINE_STAGES) {
    if (stage.stage === currentStage) {
      return completedWeight + (stageProgress / 100) * stage.weight
    }
    completedWeight += stage.weight
  }
  if (currentStage === 'complete') return 100
  if (currentStage === 'error') return 0
  return 0
}

export default function HomePage() {
  const router = useRouter()
  const [videos, setVideos] = useState<VideoFile[]>([])
  const [showDemo, setShowDemo] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<Map<string, number>>(new Map())
  const [error, setError] = useState<string | null>(null)
  const [currentStage, setCurrentStage] = useState<PipelineStatus['stage']>('idle')
  const [stageProgress, setStageProgress] = useState(0)
  const [isProcessingStarted, setIsProcessingStarted] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [eventSource, setEventSource] = useState<EventSource | null>(null)
  const [projectId, setProjectId] = useState<string | null>(null)
  const [videosDir, setVideosDir] = useState<string | null>(null)

  const abortControllerRef = useRef<AbortController | null>(null)
  const { viewerState } = usePipelineStore()

  // Calculate overall progress
  const overallProgress = calculateOverallProgress(currentStage, stageProgress)

  const handleFilesAccepted = useCallback((newFiles: VideoFile[]) => {
    setVideos(prev => [...prev, ...newFiles])
    setError(null)
  }, [])

  const handleRemoveFile = useCallback((fileId: string) => {
    setVideos(prev => prev.filter(f => f.id !== fileId))
    setUploadProgress(prev => {
      const next = new Map(prev)
      next.delete(fileId)
      return next
    })
  }, [])

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`])
  }

  // Upload files to server
  const uploadFiles = async (files: VideoFile[]): Promise<{ success: boolean; projectId?: string; videosDir?: string; error?: string }> => {
    setCurrentStage('uploading')
    setStageProgress(0)
    addLog('Starting file upload...')

    try {
      const formData = new FormData()

      // Add files
      for (const video of files) {
        if (video.preview) {
          try {
            const response = await fetch(video.preview)
            const blob = await response.blob()
            const fileExt = video.name.split('.').pop() || 'mp4'
            const idx = videos.indexOf(video)
            const fileName = `camera_${idx.toString().padStart(2, '0')}.${fileExt}`
            const file = new File([blob], fileName, { type: video.type })
            formData.append('files', file, fileName)
            addLog(`Prepared: ${fileName} (${formatFileSize(video.size)})`)
          } catch (e) {
            addLog(`Warning: Could not read ${video.name}`)
          }
        }
      }

      // Simulate progress while uploading
      let progress = 0
      const progressInterval = setInterval(() => {
        progress += Math.random() * 10
        if (progress > 95) progress = 95
        setStageProgress(Math.min(progress, 95))
      }, 200)

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      clearInterval(progressInterval)

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.message || 'Upload failed')
      }

      const data = await response.json()
      setStageProgress(100)
      addLog(`Upload complete! Project ID: ${data.projectId}`)

      return {
        success: true,
        projectId: data.projectId,
        videosDir: data.videosDir
      }
    } catch (e: any) {
      addLog(`Upload error: ${e.message}`)
      return { success: false, error: e.message }
    }
  }

  // Run the processing pipeline
  const runPipeline = async (pid: string, vDir: string) => {
    addLog('Starting 4DGS processing pipeline...')

    // Connect to SSE stream
    const eventSourceUrl = '/api/process'

    try {
      const response = await fetch(eventSourceUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projectId: pid, videosDir: vDir })
      })

      if (!response.ok) {
        throw new Error('Failed to start pipeline')
      }

      // Read the SSE stream
      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event = JSON.parse(line.slice(6))
              handlePipelineEvent(event)
            } catch (e) {
              // Skip invalid JSON
            }
          }
        }
      }

    } catch (e: any) {
      addLog(`Pipeline error: ${e.message}`)
      setCurrentStage('error')
      setError(e.message)
    }
  }

  const handlePipelineEvent = (event: any) => {
    if (event.type === 'log') {
      addLog(event.message)
    } else if (event.type === 'progress') {
      setCurrentStage(event.stage)
      setStageProgress(event.progress)
      if (event.message) {
        addLog(event.message)
      }
    } else if (event.type === 'complete') {
      setCurrentStage('complete')
      setStageProgress(100)
      addLog('🎉 Pipeline complete!')
      setTimeout(() => {
        router.push(`/viewer?projectId=${projectId}`)
      }, 2000)
    } else if (event.type === 'error') {
      setCurrentStage('error')
      setError(event.message)
      addLog(`❌ Error: ${event.message}`)
    }
  }

  const handleStartProcessing = async () => {
    if (videos.length === 0) return

    setError(null)
    setIsUploading(true)
    setCurrentStage('uploading')
    setStageProgress(0)
    setLogs([])
    setIsProcessingStarted(true)

    try {
      // Upload files first
      const uploadResult = await uploadFiles(videos)

      if (!uploadResult.success || !uploadResult.projectId || !uploadResult.videosDir) {
        setError(uploadResult.error || 'Upload failed')
        setCurrentStage('error')
        setIsUploading(false)
        return
      }

      setProjectId(uploadResult.projectId)
      setVideosDir(uploadResult.videosDir)

      // Run the pipeline
      await runPipeline(uploadResult.projectId, uploadResult.videosDir)

    } catch (e: any) {
      setError(e.message || 'An error occurred')
      setCurrentStage('error')
      addLog(`❌ Error: ${e.message}`)
      setIsUploading(false)
    }
  }

  const handleCancel = () => {
    eventSource?.close()
    setIsUploading(false)
    setIsProcessingStarted(false)
    setCurrentStage('idle')
    setStageProgress(0)
    addLog('Cancelled by user')
    router.push('/')
  }

  return (
    <div className="min-h-screen">
      {/* Hero section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600" />
        <div className="absolute inset-0 bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%23ffffff%22%20fill-opacity%3D%220.05%22%3E%3Cpath%20d%3D%22M36%2034v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6%2034v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6%204V0H4v4H0v2h4v4h2V6h4V4H6z%22%2F%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')] opacity-30" />

        <div className="relative container mx-auto px-4 py-16">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
              Transform Video into
              <span className="block bg-gradient-to-r from-yellow-300 to-orange-400 bg-clip-text text-transparent">
                4D Gaussian Splats
              </span>
            </h1>
            <p className="text-lg text-white/80">
              Upload multi-view video sequences for dynamic 3D reconstruction
            </p>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 120" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M0 120L60 105C120 90 240 60 360 45C480 30 600 30 720 37.5C840 45 960 60 1080 67.5C1200 75 1320 75 1380 75L1440 75V120H1380C1320 120 1200 120 1080 120C960 120 840 120 720 120C600 120 480 120 360 120C240 120 120 120 60 120H0Z" fill="currentColor" className="text-gray-50 dark:text-gray-900" />
          </svg>
        </div>
      </section>

      {/* Processing Status Section */}
      {isProcessingStarted && (
        <section className="container mx-auto px-4 py-8 -mt-8">
          <div className="max-w-4xl mx-auto bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
            {/* Header */}
            <div className={`px-6 py-4 ${currentStage === 'error' ? 'bg-red-600' : currentStage === 'complete' ? 'bg-green-600' : 'bg-gradient-to-r from-blue-600 to-purple-600'}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {currentStage === 'error' ? (
                    <AlertCircle className="w-8 h-8 text-red-200" />
                  ) : currentStage === 'complete' ? (
                    <CheckCircle className="w-8 h-8 text-green-200" />
                  ) : (
                    <Loader2 className="w-8 h-8 text-white animate-spin" />
                  )}
                  <div>
                    <h2 className="text-xl font-bold text-white">
                      {currentStage === 'error' ? 'Error' : currentStage === 'complete' ? 'Complete!' : 'Processing...'}
                    </h2>
                    <p className="text-white/80 text-sm">
                      {currentStage === 'error' ? error : currentStage === 'complete' ? 'Your 4D model is ready!' : 'Please wait while we process your videos'}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-4xl font-bold text-white">{Math.round(overallProgress)}%</span>
                </div>
              </div>
            </div>

            {/* Overall progress bar */}
            <div className="h-3 bg-gray-100 dark:bg-gray-900">
              <div
                className={`h-full transition-all duration-500 ease-out relative ${currentStage === 'error' ? 'bg-red-500' : currentStage === 'complete' ? 'bg-green-500' : 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500'}`}
                style={{ width: `${overallProgress}%` }}
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
              </div>
            </div>

            {/* Pipeline stages */}
            <div className="p-6 space-y-4">
              {PIPELINE_STAGES.map((stageConfig, index) => {
                const Icon = stageConfig.icon
                const isComplete = PIPELINE_STAGES.findIndex(s => s.stage === currentStage) > index ||
                  currentStage === 'complete'
                const isCurrent = currentStage === stageConfig.stage
                const isError = currentStage === 'error' && index === PIPELINE_STAGES.length - 1

                return (
                  <div key={stageConfig.stage} className="flex items-center gap-4">
                    {/* Stage indicator */}
                    <div className={`
                      w-10 h-10 rounded-full flex items-center justify-center
                      ${isComplete || currentStage === 'complete' ? 'bg-green-500 text-white' : ''}
                      ${isCurrent && currentStage !== 'complete' && currentStage !== 'error' ? 'bg-blue-500 text-white' : ''}
                      ${isError ? 'bg-red-500 text-white' : ''}
                      ${!isComplete && !isCurrent && !isError ? 'bg-gray-200 dark:bg-gray-700 text-gray-400' : ''}
                      transition-colors duration-300
                    `}>
                      {isComplete || currentStage === 'complete' ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : (
                        <Icon className={`w-5 h-5 ${isCurrent && currentStage !== 'complete' ? 'animate-pulse' : ''}`} />
                      )}
                    </div>

                    {/* Stage info */}
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className={`font-medium ${isCurrent ? 'text-blue-600 dark:text-blue-400' : ''}`}>
                          {stageConfig.label}
                        </span>
                        <span className="text-sm text-gray-500">
                          {stageConfig.weight}%
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {stageConfig.description}
                      </p>

                      {/* Stage progress bar */}
                      {isCurrent && currentStage !== 'error' && currentStage !== 'complete' && (
                        <div className="mt-2 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 transition-all duration-300"
                            style={{ width: `${stageProgress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Logs */}
            <div className="border-t border-gray-200 dark:border-gray-700">
              <div className="px-6 py-3 bg-gray-50 dark:bg-gray-900 flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Live Logs</span>
                <button
                  onClick={() => setLogs([])}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  Clear
                </button>
              </div>
              <div className="h-48 overflow-y-auto p-4 bg-gray-900 font-mono text-xs text-green-400">
                {logs.length === 0 ? (
                  <span className="text-gray-500">Waiting for logs...</span>
                ) : (
                  logs.map((log, i) => (
                    <div key={i} className="py-0.5">{log}</div>
                  ))
                )}
              </div>
            </div>

            {/* Cancel/Back button */}
            {currentStage !== 'complete' && (
              <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700">
                <button
                  onClick={handleCancel}
                  className="w-full py-2 px-4 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors font-medium"
                >
                  Cancel Processing
                </button>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Upload section */}
      {!isProcessingStarted && (
        <section className="container mx-auto px-4 py-8">
          <div className="max-w-5xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 p-8">
              <div className="mb-6">
                <h2 className="text-2xl font-bold mb-2">Upload Your Videos</h2>
                <p className="text-gray-500 dark:text-gray-400">
                  Drop synchronized multi-view video files to begin 4D reconstruction
                </p>
              </div>

              <DropZone onFilesAccepted={handleFilesAccepted} maxFiles={8} />

              {/* Error display */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <span className="font-medium text-red-700 dark:text-red-400">Error</span>
                  </div>
                  <p className="mt-2 text-sm text-red-600 dark:text-red-300">{error}</p>
                </div>
              )}

              {videos.length > 0 && (
                <FileList files={videos} onRemove={handleRemoveFile} />
              )}

              {videos.length > 0 && (
                <div className="mt-8 flex justify-center">
                  <button
                    onClick={handleStartProcessing}
                    className="flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg bg-gradient-to-r from-blue-600 to-purple-600 text-white hover:shadow-lg hover:scale-105 transition-all duration-200"
                  >
                    <Play className="w-5 h-5" />
                    Start 4D Reconstruction
                  </button>
                </div>
              )}

              {videos.length > 0 && videos.length < 3 && (
                <p className="mt-4 text-center text-sm text-amber-600 dark:text-amber-400">
                  ⚠️ Minimum 3 camera views recommended for accurate 4D reconstruction
                </p>
              )}
            </div>

            {/* Demo viewer */}
            <div className="mt-8 text-center">
              <button
                onClick={() => setShowDemo(!showDemo)}
                className="text-blue-500 hover:text-blue-600 text-sm underline"
              >
                {showDemo ? 'Hide' : 'Show'} demo 4D Gaussian viewer
              </button>
            </div>

            {showDemo && (
              <div className="mt-8">
                {(() => {
                  const Viewer3D = dynamic(() => import('@/components/Viewer3D').then(mod => mod.Viewer3D), { ssr: false })
                  return <Viewer3D viewerState={viewerState} className="h-[500px] rounded-xl overflow-hidden" />
                })()}
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  )
}