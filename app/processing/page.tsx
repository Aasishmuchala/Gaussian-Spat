'use client'

import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import { ProcessingProgress } from '@/components/ProcessingProgress'
import { usePipelineStore } from '@/lib/store'
import type { PipelineStatus } from '@/lib/types'
import { ArrowLeft, ExternalLink } from 'lucide-react'

const Viewer3D = dynamic(
  () => import('@/components/Viewer3D').then(mod => mod.Viewer3D),
  { ssr: false }
)

export default function ProcessingPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const projectId = searchParams.get('projectId')

  const { project, viewerState, updateStatus } = usePipelineStore()
  const [status, setStatus] = useState<PipelineStatus>({
    stage: 'extracting_frames',
    progress: 0,
    message: 'Initializing...'
  })
  const [logs, setLogs] = useState<string[]>([])
  const [showViewer, setShowViewer] = useState(false)

  useEffect(() => {
    if (!projectId) {
      router.push('/')
      return
    }

    // Simulate pipeline progress
    const stages: PipelineStatus['stage'][] = [
      'extracting_frames',
      'running_colmap',
      'training_4dgs',
      'starting_viewer'
    ]

    let currentStageIndex = 0
    let progress = 0

    const interval = setInterval(() => {
      progress += Math.random() * 3

      if (progress >= 100) {
        progress = 0
        currentStageIndex++

        if (currentStageIndex >= stages.length) {
          setStatus({
            stage: 'complete',
            progress: 100,
            message: 'Processing complete!'
          })
          setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Pipeline completed successfully!`])
          clearInterval(interval)
          return
        }

        const stage = stages[currentStageIndex]
        setStatus(prev => ({
          ...prev,
          stage,
          progress: 0,
          message: getStageMessage(stage)
        }))

        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Started: ${stage.replace('_', ' ')}`])
      }

      setStatus(prev => ({
        ...prev,
        progress: Math.min(progress, 99)
      }))

      // Random log entries
      if (Math.random() > 0.9) {
        const logMessages = [
          'Processing frame 100/500',
          'Feature extraction: 2341 points',
          'Triangulation: 1892 valid points',
          'Gaussian initialization complete',
          'Iteration 500/30000',
          'Loss: 0.0234',
          'Densification triggered',
          'Memory usage: 4.2GB / 8GB'
        ]
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${logMessages[Math.floor(Math.random() * logMessages.length)]}`])
      }
    }, 500)

    return () => clearInterval(interval)
  }, [projectId, router])

  const handleComplete = () => {
    setShowViewer(true)
  }

  const handleGoBack = () => {
    router.push('/')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <button
            onClick={handleGoBack}
            className="flex items-center gap-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Upload
          </button>

          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-blue-500 animate-pulse" />
            <span className="text-sm text-gray-500">Processing your 4D reconstruction</span>
          </div>

          {status.stage === 'complete' && (
            <button
              onClick={() => setShowViewer(!showViewer)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              {showViewer ? 'Hide Viewer' : 'View 4D Model'}
            </button>
          )}
        </div>

        {/* Main content */}
        <div className="max-w-6xl mx-auto">
          {/* Progress section */}
          <div className="mb-8">
            <ProcessingProgress status={status} onComplete={handleComplete} />
          </div>

          {/* Two column layout */}
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Logs panel */}
            <div className="lg:col-span-1">
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="font-semibold">Processing Logs</h3>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Real-time pipeline output
                  </p>
                </div>
                <div className="h-80 overflow-y-auto p-4 font-mono text-xs bg-gray-50 dark:bg-gray-900">
                  {logs.map((log, i) => (
                    <div key={i} className="text-gray-600 dark:text-gray-400 py-0.5">
                      {log}
                    </div>
                  ))}
                  {logs.length === 0 && (
                    <div className="text-gray-400 dark:text-gray-600 italic">
                      Waiting for pipeline output...
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Viewer section */}
            <div className="lg:col-span-2">
              {showViewer && status.stage === 'complete' ? (
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="font-semibold">4D Gaussian Splat Viewer</h3>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Interactive temporal visualization
                    </p>
                  </div>
                  <Viewer3D viewerState={viewerState} className="h-96" />
                </div>
              ) : (
                <div className="h-96 bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 flex items-center justify-center">
                  <div className="text-center text-gray-400 dark:text-gray-600">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    </div>
                    <p>4D Viewer will appear here when processing is complete</p>
                  </div>
                </div>
              )}

              {/* Info cards */}
              <div className="mt-6 grid sm:grid-cols-2 gap-4">
                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <h4 className="font-medium mb-2">Processing Info</h4>
                  <dl className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Project ID:</dt>
                      <dd className="font-mono text-xs">{projectId?.slice(0, 8)}...</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">GPU Memory:</dt>
                      <dd>4.2 GB / 8 GB</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Gaussians:</dt>
                      <dd>~2.4M</dd>
                    </div>
                  </dl>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                  <h4 className="font-medium mb-2">Estimated Time</h4>
                  <dl className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-gray-500">Frame Extraction:</dt>
                      <dd>~30s</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">COLMAP SfM:</dt>
                      <dd>~5-15 min</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-gray-500">4DGS Training:</dt>
                      <dd>~10-30 min</dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function getStageMessage(stage: PipelineStatus['stage']): string {
  const messages: Record<PipelineStatus['stage'], string> = {
    idle: 'Ready',
    uploading: 'Uploading videos...',
    extracting_frames: 'Extracting frames from video files...',
    running_colmap: 'Running COLMAP Structure-from-Motion...',
    training_4dgs: 'Training 4D Gaussian Splatting model...',
    starting_viewer: 'Starting interactive viewer...',
    complete: 'Complete!',
    error: 'Error occurred'
  }
  return messages[stage]
}