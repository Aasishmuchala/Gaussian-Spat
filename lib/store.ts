import { create } from 'zustand'
import type { Project, PipelineStatus, VideoFile, ViewerState } from './types'

interface PipelineStore {
  // Current project
  project: Project | null

  // Viewer state
  viewerState: ViewerState

  // Actions
  setProject: (project: Project | null) => void
  updateStatus: (status: Partial<PipelineStatus>) => void
  addVideos: (videos: VideoFile[]) => void
  removeVideo: (videoId: string) => void
  updateVideoStatus: (videoId: string, status: VideoFile['status']) => void

  // Viewer actions
  setCurrentTime: (time: number) => void
  setIsPlaying: (playing: boolean) => void
  setCameraPosition: (position: [number, number, number]) => void
  setTemporalThreshold: (threshold: number) => void

  // Reset
  resetProject: () => void
}

const initialViewerState: ViewerState = {
  currentTime: 0,
  totalFrames: 60,
  isPlaying: false,
  cameraPosition: [0, 0, 5],
  cameraTarget: [0, 0, 0],
  showTrajectory: true,
  showVelocity: false,
  temporalThreshold: 0.05,
}

export const usePipelineStore = create<PipelineStore>((set) => ({
  project: null,
  viewerState: initialViewerState,

  setProject: (project) => set({ project }),

  updateStatus: (statusUpdate) => set((state) => {
    if (!state.project) return state
    return {
      project: {
        ...state.project,
        status: { ...state.project.status, ...statusUpdate }
      }
    }
  }),

  addVideos: (videos) => set((state) => {
    if (!state.project) return state
    return {
      project: {
        ...state.project,
        videos: [...state.project.videos, ...videos]
      }
    }
  }),

  removeVideo: (videoId) => set((state) => {
    if (!state.project) return state
    return {
      project: {
        ...state.project,
        videos: state.project.videos.filter(v => v.id !== videoId)
      }
    }
  }),

  updateVideoStatus: (videoId, status) => set((state) => {
    if (!state.project) return state
    return {
      project: {
        ...state.project,
        videos: state.project.videos.map(v =>
          v.id === videoId ? { ...v, status } : v
        )
      }
    }
  }),

  setCurrentTime: (time) => set((state) => ({
    viewerState: { ...state.viewerState, currentTime: time }
  })),

  setIsPlaying: (playing) => set((state) => ({
    viewerState: { ...state.viewerState, isPlaying: playing }
  })),

  setCameraPosition: (position) => set((state) => ({
    viewerState: { ...state.viewerState, cameraPosition: position }
  })),

  setTemporalThreshold: (threshold) => set((state) => ({
    viewerState: { ...state.viewerState, temporalThreshold: threshold }
  })),

  resetProject: () => set({ project: null, viewerState: initialViewerState }),
}))

// Utility functions
export function generateId(): string {
  return Math.random().toString(36).substring(2, 15)
}

export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

export function getStageLabel(stage: PipelineStatus['stage']): string {
  const labels: Record<PipelineStatus['stage'], string> = {
    idle: 'Ready',
    uploading: 'Uploading Videos',
    extracting_frames: 'Extracting Frames',
    running_colmap: 'Running COLMAP SfM',
    training_4dgs: 'Training 4D Gaussians',
    starting_viewer: 'Starting Viewer',
    complete: 'Complete',
    error: 'Error',
  }
  return labels[stage]
}

export function getStageColor(stage: PipelineStatus['stage']): string {
  const colors: Record<PipelineStatus['stage'], string> = {
    idle: 'bg-gray-500',
    uploading: 'bg-blue-500',
    extracting_frames: 'bg-cyan-500',
    running_colmap: 'bg-yellow-500',
    training_4dgs: 'bg-purple-500',
    starting_viewer: 'bg-green-500',
    complete: 'bg-green-600',
    error: 'bg-red-500',
  }
  return colors[stage]
}