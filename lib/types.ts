// Pipeline types
export type PipelineStage =
  | 'idle'
  | 'uploading'
  | 'extracting_frames'
  | 'running_colmap'
  | 'training_4dgs'
  | 'starting_viewer'
  | 'complete'
  | 'error'

export interface PipelineStatus {
  stage: PipelineStage
  progress: number // 0-100
  message: string
  currentStep?: string
  totalSteps?: number
  elapsedTime?: number // seconds
  estimatedTimeRemaining?: number // seconds
  error?: string
}

export interface VideoFile {
  id: string
  name: string
  size: number
  type: string
  preview?: string
  status: 'pending' | 'uploading' | 'uploaded' | 'processing' | 'complete' | 'error'
}

export interface Project {
  id: string
  name: string
  videos: VideoFile[]
  status: PipelineStatus
  createdAt: Date
  outputPath?: string
  checkpointPath?: string
  viewerUrl?: string
}

// Gaussian splat types
export interface GaussianData {
  positions: Float32Array
  colors: Float32Array
  scales?: Float32Array
  rotations?: Float32Array
  opacities?: Float32Array
}

export interface ViewerState {
  currentTime: number
  totalFrames: number
  isPlaying: boolean
  cameraPosition: [number, number, number]
  cameraTarget: [number, number, number]
  showTrajectory: boolean
  showVelocity: boolean
  temporalThreshold: number
}

// API types
export interface UploadResponse {
  success: boolean
  projectId: string
  message?: string
}

export interface ProcessResponse {
  success: boolean
  projectId: string
  status: PipelineStatus
}

export interface StreamEvent {
  type: 'status' | 'log' | 'error' | 'complete'
  data: PipelineStatus | string
}