'use client'

import { Suspense, useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Environment, Grid, Html } from '@react-three/drei'
import * as THREE from 'three'
import { useSearchParams } from 'next/navigation'
import { ViewerState } from '@/lib/types'

interface Viewer3DProps {
  viewerState: ViewerState
  className?: string
  onTimeChange?: (time: number) => void
}

// Point cloud renderer for Gaussian splats
function GaussianPoints({
  positions,
  colors,
  currentTime,
  temporalThreshold,
  isPlaying,
  onTimeChange
}: {
  positions: Float32Array
  colors: Float32Array
  currentTime: number
  temporalThreshold: number
  isPlaying: boolean
  onTimeChange?: (time: number) => void
}) {
  const pointsRef = useRef<THREE.Points>(null)
  const timeRef = useRef(currentTime)

  // Animation for playback
  useFrame((_, delta) => {
    if (isPlaying && onTimeChange) {
      timeRef.current += delta * 10 // 10 frames per second
      if (timeRef.current >= 60) timeRef.current = 0
      onTimeChange(timeRef.current)
    }

    if (pointsRef.current) {
      pointsRef.current.rotation.y += 0.001
    }
  })

  // Create geometry from typed arrays
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return geo
  }, [positions, colors])

  // Custom shader material for Gaussians
  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.9,
      sizeAttenuation: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  }, [])

  return (
    <points ref={pointsRef} geometry={geometry} material={material} />
  )
}

// Camera controller
function CameraController({
  cameraPosition
}: {
  cameraPosition: [number, number, number]
}) {
  const { camera } = useThree()

  useFrame(() => {
    camera.position.lerp(new THREE.Vector3(...cameraPosition), 0.05)
    camera.lookAt(0, 0, 0)
  })

  return (
    <PerspectiveCamera
      makeDefault
      position={cameraPosition}
      fov={50}
      near={0.1}
      far={1000}
    />
  )
}

// Loading fallback
function LoadingFallback() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-4">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-white text-sm">Loading 4D Gaussian Splat...</p>
      </div>
    </Html>
  )
}

// Demo data generator
function generateDemoData(numPoints: number = 50000) {
  const positions = new Float32Array(numPoints * 3)
  const colors = new Float32Array(numPoints * 3)

  for (let i = 0; i < numPoints; i++) {
    // Create a more interesting shape - multiple spheres with color gradients
    const shapeType = Math.random()
    let x, y, z

    if (shapeType < 0.7) {
      // Main sphere with noise
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 2 + Math.random() * 0.3
      x = r * Math.sin(phi) * Math.cos(theta)
      y = r * Math.sin(phi) * Math.sin(theta) + Math.sin(Math.random() * Math.PI) * 0.5
      z = r * Math.cos(phi)
    } else if (shapeType < 0.9) {
      // Floating particles
      x = (Math.random() - 0.5) * 4
      y = (Math.random() - 0.5) * 4
      z = (Math.random() - 0.5) * 4
    } else {
      // Ring structure
      const theta = Math.random() * Math.PI * 2
      const r = 3 + Math.random() * 0.2
      x = r * Math.cos(theta)
      y = (Math.random() - 0.5) * 0.5
      z = r * Math.sin(theta)
    }

    positions[i * 3] = x
    positions[i * 3 + 1] = y
    positions[i * 3 + 2] = z

    // Color based on position and height
    const hue = (x + 3) / 6 + 0.5
    const sat = 0.6 + Math.random() * 0.3
    const light = 0.4 + (y + 3) / 12

    // Convert HSV to RGB
    const h = hue % 1
    const s = Math.min(1, sat)
    const l = Math.min(1, light)

    const c = (1 - Math.abs(2 * l - 1)) * s
    const x2 = c * (1 - Math.abs((h * 6) % 2 - 1))
    const m = l - c / 2

    let r2, g, b
    if (h < 1/6) { r2 = c; g = x2; b = 0 }
    else if (h < 2/6) { r2 = x2; g = c; b = 0 }
    else if (h < 3/6) { r2 = 0; g = c; b = x2 }
    else if (h < 4/6) { r2 = 0; g = x2; b = c }
    else if (h < 5/6) { r2 = x2; g = 0; b = c }
    else { r2 = c; g = 0; b = x2 }

    colors[i * 3] = r2 + m
    colors[i * 3 + 1] = g + m
    colors[i * 3 + 2] = b + m
  }

  return { positions, colors }
}

// Main viewer component
export function Viewer3D({
  viewerState,
  className = '',
  onTimeChange
}: Viewer3DProps) {
  const [demoData, setDemoData] = useState<{ positions: Float32Array; colors: Float32Array } | null>(null)
  const searchParams = useSearchParams()

  useEffect(() => {
    // Generate demo data on mount
    const data = generateDemoData
    setDemoData(data)
  }, [])

  const positions = demoData?.positions || new Float32Array(0)
  const colors = demoData?.colors || new Float32Array(0)

  return (
    <div className={`relative ${className}`}>
      <Canvas
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: 'high-performance'
        }}
        className="bg-gradient-to-b from-gray-900 to-gray-950"
      >
        <Suspense fallback={<LoadingFallback />}>
          {/* Camera */}
          <CameraController cameraPosition={viewerState.cameraPosition} />

          {/* Controls */}
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            dampingFactor={0.05}
          />

          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1.2} />
          <pointLight position={[-10, 5, -10]} intensity={0.6} color="#8844ff" />

          {/* Grid helper */}
          <Grid
            args={[30, 30]}
            cellSize={0.5}
            cellThickness={0.5}
            cellColor="#222222"
            sectionSize={2}
            sectionThickness={1}
            sectionColor="#444444"
            fadeDistance={50}
            fadeStrength={1}
            followCamera={false}
            infiniteGrid={true}
          />

          {/* Gaussian points */}
          {demoData && (
            <GaussianPoints
              positions={positions}
              colors={colors}
              currentTime={viewerState.currentTime}
              temporalThreshold={viewerState.temporalThreshold}
              isPlaying={viewerState.isPlaying}
              onTimeChange={onTimeChange}
            />
          )}

          {/* Environment */}
          <Environment preset="night" />
        </Suspense>
      </Canvas>

      {/* Title overlay */}
      <div className="absolute top-4 left-4 bg-black/80 backdrop-blur-sm rounded-lg px-4 py-2">
        <h3 className="text-white font-medium">4D Gaussian Splat Viewer</h3>
        <p className="text-white/60 text-xs">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </p>
      </div>
    </div>
  )
}

// Dynamic wrapper for SSR
export default function Viewer3DDynamic(props: Viewer3DProps) {
  return (
    <Suspense
      fallback={
        <div className="w-full h-full flex items-center justify-center bg-gray-900">
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-white text-sm">Loading 3D Viewer...</p>
          </div>
        </div>
      }
    >
      <Viewer3D {...props} />
    </Suspense>
  )
}