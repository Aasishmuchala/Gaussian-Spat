'use client'

import dynamic from 'next/dynamic'
import { usePipelineStore } from '@/lib/store'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'

const Viewer3D = dynamic(
  () => import('@/components/Viewer3D').then(mod => mod.Viewer3D),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full flex items-center justify-center bg-gray-900">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-white text-sm">Loading 3D Viewer...</p>
        </div>
      </div>
    )
  }
)

export default function ViewerPage() {
  const searchParams = useSearchParams()
  const projectId = searchParams.get('projectId')

  const { viewerState, setCurrentTime, setIsPlaying, setTemporalThreshold } = usePipelineStore()

  return (
    <div className="h-screen flex flex-col bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            ← Back
          </Link>
          <div className="w-px h-6 bg-gray-600" />
          <h1 className="text-white font-semibold">4D Gaussian Splat Viewer</h1>
          {projectId && (
            <span className="text-xs text-gray-500 font-mono">
              Project: {projectId.slice(0, 8)}...
            </span>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="text-sm text-gray-400">
            Temporal: {viewerState.temporalThreshold.toFixed(2)}
          </div>
          <div className="text-sm text-gray-400">
            Time: {Math.floor(viewerState.currentTime).toString().padStart(3, '0')} / {viewerState.totalFrames}
          </div>
        </div>
      </div>

      {/* 3D Viewer */}
      <div className="flex-1 relative">
        <Viewer3D viewerState={viewerState} className="w-full h-full" />

        {/* Controls overlay */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 bg-black/80 backdrop-blur-sm rounded-full px-6 py-3">
          <button
            onClick={() => setIsPlaying(!viewerState.isPlaying)}
            className="w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors"
          >
            {viewerState.isPlaying ? (
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <polygon points="5,3 19,12 5,21" />
              </svg>
            )}
          </button>

          <div className="flex items-center gap-3">
            <span className="text-white text-sm font-mono w-12 text-right">
              {Math.floor(viewerState.currentTime).toString().padStart(3, '0')}
            </span>

            <input
              type="range"
              min={0}
              max={viewerState.totalFrames - 1}
              value={viewerState.currentTime}
              onChange={(e) => setCurrentTime(Number(e.target.value))}
              className="w-64 h-2 bg-white/20 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:w-4
                [&::-webkit-slider-thumb]:h-4
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-blue-500
                [&::-webkit-slider-thumb]:cursor-pointer
                [&::-webkit-slider-thumb]:shadow-lg"
            />

            <span className="text-white text-sm font-mono w-12">
              {viewerState.totalFrames.toString().padStart(3, '0')}
            </span>
          </div>
        </div>

        {/* Settings panel */}
        <div className="absolute top-4 right-4 bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3 space-y-3">
          <div>
            <label className="text-white text-xs font-medium block mb-1">
              Temporal Threshold: {viewerState.temporalThreshold.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={viewerState.temporalThreshold}
              onChange={(e) => setTemporalThreshold(Number(e.target.value))}
              className="w-32 h-1.5 bg-white/20 rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:w-3
                [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-blue-500
                [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>

          <div className="flex gap-2">
            <button
              className={`px-2 py-1 text-xs rounded transition-colors ${
                viewerState.showTrajectory ? 'bg-blue-500 text-white' : 'bg-white/10 text-white/70'
              }`}
            >
              Trajectory
            </button>
            <button
              className={`px-2 py-1 text-xs rounded transition-colors ${
                viewerState.showVelocity ? 'bg-purple-500 text-white' : 'bg-white/10 text-white/70'
              }`}
            >
              Velocity
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}