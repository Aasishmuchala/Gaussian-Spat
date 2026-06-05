'use client'

import { useEffect, useRef } from 'react'
import { PipelineStatus } from '@/lib/types'
import { getStageLabel, getStageColor, formatTime } from '@/lib/store'
import {
  Upload,
  Scissors,
  Camera,
  Cpu,
  Eye,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react'

interface ProcessingProgressProps {
  status: PipelineStatus
  onComplete?: () => void
}

const STAGE_ICONS = {
  idle: Upload,
  uploading: Upload,
  extracting_frames: Scissors,
  running_colmap: Camera,
  training_4dgs: Cpu,
  starting_viewer: Eye,
  complete: CheckCircle,
  error: AlertCircle,
}

const STAGE_DESCRIPTIONS = {
  idle: 'Ready to process',
  uploading: 'Uploading your videos...',
  extracting_frames: 'Extracting frames from each camera angle using FFmpeg',
  running_colmap: 'Running COLMAP Structure-from-Motion to reconstruct 3D points',
  training_4dgs: 'Training 4D Gaussian Splatting model - this may take several minutes',
  starting_viewer: 'Starting the interactive 4D viewer server',
  complete: 'Processing complete! Your 4D model is ready.',
  error: 'An error occurred during processing',
}

export function ProcessingProgress({ status, onComplete }: ProcessingProgressProps) {
  const hasCompletedRef = useRef(false)

  useEffect(() => {
    if (status.stage === 'complete' && !hasCompletedRef.current) {
      hasCompletedRef.current = true
      onComplete?.()
    }
    if (status.stage !== 'complete') {
      hasCompletedRef.current = false
    }
  }, [status.stage, onComplete])

  const Icon = STAGE_ICONS[status.stage]
  const colorClass = getStageColor(status.stage)
  const description = STAGE_DESCRIPTIONS[status.stage]

  // Calculate overall progress
  const stageWeights = {
    uploading: 10,
    extracting_frames: 20,
    running_colmap: 30,
    training_4dgs: 35,
    starting_viewer: 5,
    complete: 100,
    error: 0,
    idle: 0,
  }

  const overallProgress = stageWeights[status.stage] * (status.progress / 100)

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Progress card */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        {/* Header */}
        <div className={`px-6 py-4 ${colorClass} text-white`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <Icon className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">{getStageLabel(status.stage)}</h2>
                <p className="text-sm opacity-90">{description}</p>
              </div>
            </div>

            {/* Progress percentage */}
            <div className="text-right">
              <span className="text-3xl font-bold">{Math.round(status.progress)}%</span>
              {status.estimatedTimeRemaining && status.stage !== 'complete' && (
                <p className="text-xs opacity-80">
                  ~{formatTime(status.estimatedTimeRemaining)} remaining
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div className="h-2 bg-gray-100 dark:bg-gray-900">
          <div
            className={`h-full ${colorClass} transition-all duration-500 ease-out relative overflow-hidden`}
            style={{ width: `${status.progress}%` }}
          >
            {/* Shimmer effect */}
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
          </div>
        </div>

        {/* Details */}
        <div className="p-6 space-y-4">
          {/* Current step */}
          {status.currentStep && (
            <div className="flex items-start gap-3">
              <Loader2 className="w-5 h-5 text-blue-500 animate-spin flex-shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {status.currentStep}
                </p>
                {status.totalSteps && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Step {Math.min(status.totalSteps, status.totalSteps)} of {status.totalSteps}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Error message */}
          {status.error && (
            <div className="p-4 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-700 dark:text-red-400">
                    Error Details
                  </p>
                  <p className="text-sm text-red-600 dark:text-red-300 mt-1 whitespace-pre-wrap">
                    {status.error}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Time elapsed */}
          {status.elapsedTime !== undefined && status.stage !== 'complete' && (
            <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
              <span>Elapsed time</span>
              <span className="font-mono">{formatTime(status.elapsedTime)}</span>
            </div>
          )}

          {/* Pipeline stages visualization */}
          <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3">
              Pipeline Stages
            </p>
            <div className="flex items-center gap-1">
              {['extracting_frames', 'running_colmap', 'training_4dgs', 'starting_viewer'].map((stage, index) => {
                const isComplete = ['uploading', 'extracting_frames', 'running_colmap', 'training_4dgs', 'starting_viewer', 'complete']
                  .indexOf(status.stage) > index
                const isActive = stage === status.stage

                return (
                  <div key={stage} className="flex items-center">
                    <div
                      className={`
                        flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium
                        transition-colors duration-200
                        ${isComplete ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : ''}
                        ${isActive ? `${colorClass} text-white` : ''}
                        ${!isComplete && !isActive ? 'bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-500' : ''}
                      `}
                    >
                      {isComplete && <CheckCircle className="w-3 h-3" />}
                      {isActive && <Loader2 className="w-3 h-3 animate-spin" />}
                      {getStageLabel(stage as any).replace('Running ', '').replace('Extracting ', '').replace('Training ', '').replace('Starting ', '')}
                    </div>
                    {index < 3 && (
                      <div className={`w-4 h-0.5 ${isComplete ? 'bg-green-300 dark:bg-green-700' : 'bg-gray-200 dark:bg-gray-700'}`} />
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}