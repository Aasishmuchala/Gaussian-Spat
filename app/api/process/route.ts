import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'

// Store active processes
const activeProcesses = new Map<string, { process: any; startTime: number }>()

// Pipeline stage definitions with weights
const PIPELINE_STAGES = {
  extracting_frames: { weight: 15, message: 'Extracting frames from video...' },
  running_colmap: { weight: 30, message: 'Running multi-view 3D reconstruction...' },
  processing_keyframes: { weight: 20, message: 'Processing keyframes with velocity estimation...' },
  training_4dgs: { weight: 30, message: 'Training 4D Gaussian Splatting...' },
  complete: { weight: 5, message: 'Finalizing results...' },
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { projectId, videosDir } = body

    if (!projectId) {
      return NextResponse.json(
        { success: false, message: 'Project ID required' },
        { status: 400 }
      )
    }

    // Determine the output directory
    const outputPath = path.join(process.cwd(), 'uploads', projectId)

    // Get video files from the project directory
    let videoFiles: string[] = []
    const projectDir = path.join(process.cwd(), 'uploads', projectId)

    if (fs.existsSync(projectDir)) {
      const files = fs.readdirSync(projectDir)
      videoFiles = files
        .filter(f => /\.(mp4|mov|avi|webm|mkv)$/i.test(f))
        .map(f => path.join(projectDir, f))
    }

    if (videoFiles.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No video files found' },
        { status: 400 }
      )
    }

    console.log('Starting 4DGS pipeline for videos:', videoFiles)

    // Create streaming response
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        let isControllerClosed = false
        const closeController = () => {
          if (!isControllerClosed) {
            isControllerClosed = true
            try { controller.close() } catch (e) { /* already closed */ }
          }
        }

        const sendEvent = (type: string, data: any) => {
          try {
            const jsonData = { type, ...data }
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(jsonData)}\n\n`))
          } catch (e) {
            closeController()
          }
        }

        const startTime = Date.now()

        // Build the Python command
        const pythonPath = path.join('C:', 'Users', 'Sthyra', 'freetime_env', 'Scripts', 'python.exe')
        const pipelineScript = path.join(process.cwd(), 'lib', 'pipeline', 'main_pipeline.py')

        // Build args - use comma-separated video paths
        const videoArg = videoFiles.join(',')
        const cmdArgs = [
          pipelineScript,
          '--videos', videoArg,
          '--output-dir', outputPath,
          '--fps', '2',
          '--max-frames', '60',
          '--max-steps', '5000',
          '--num-points', '100000'
        ]

        sendEvent('log', { message: `🚀 4DGS Pipeline starting with ${videoFiles.length} videos...` })
        sendEvent('log', { message: `📁 Output: ${outputPath}` })
        sendEvent('log', { message: '─'.repeat(50) })

        let currentStage = 'setup'
        let pythonProcess: any = null

        try {
          pythonProcess = spawn(pythonPath, cmdArgs, {
            cwd: process.cwd(),
            shell: true,
            windowsHide: true
          })

          activeProcesses.set(projectId, {
            process: pythonProcess,
            startTime
          })

          // Parse stdout for progress
          pythonProcess.stdout.on('data', (data: Buffer) => {
            const output = data.toString()
            console.log('[Python]', output.trim())

            // Parse progress from line format: [XX.X%] message
            const lines = output.split('\n')
            for (const line of lines) {
              if (!line.trim()) continue

              const match = line.match(/\[(\d+\.?\d*)%\]\s*(.*)/)
              if (match) {
                const overallProgress = parseFloat(match[1])
                const message = match[2]

                // Determine stage from message
                let stage = 'setup'
                let progressInStage = 0

                if (message.includes('Extracting')) {
                  stage = 'extracting_frames'
                  progressInStage = overallProgress
                }
                else if (message.includes('Reconstructing') || message.includes('point cloud') || message.includes('Running COLMAP')) {
                  stage = 'running_colmap'
                  progressInStage = overallProgress
                }
                else if (message.includes('Processing keyframe') || message.includes('keyframes')) {
                  stage = 'processing_keyframes'
                  progressInStage = overallProgress
                }
                else if (message.includes('Training') || message.includes('4DGS') || message.includes('Gaussians')) {
                  stage = 'training_4dgs'
                  progressInStage = overallProgress
                }
                else if (message.includes('complete') || message.includes('success') || message.includes('saved to')) {
                  stage = 'complete'
                  progressInStage = 100
                }

                sendEvent('progress', {
                  stage,
                  progress: progressInStage,
                  message: message.trim(),
                  elapsedTime: Math.floor((Date.now() - startTime) / 1000)
                })

                currentStage = stage
              } else {
                // Pass through log messages
                if (line.trim() && !line.startsWith('[')) {
                  sendEvent('log', { message: line.trim() })
                }
              }
            }
          })

          pythonProcess.stderr.on('data', (data: Buffer) => {
            const message = data.toString().trim()
            if (message && message.length > 2 && !message.match(/^\s{2,}/)) {
              console.error('[Python Error]', message)
              sendEvent('log', { message: `[ERR] ${message}` })
            }
          })

          pythonProcess.on('close', (code: number | null) => {
            activeProcesses.delete(projectId)

            const elapsed = Math.floor((Date.now() - startTime) / 1000)

            if (code === 0) {
              sendEvent('complete', {
                message: '🎉 Pipeline completed successfully!',
                elapsedTime: elapsed,
                outputDir: outputPath,
                viewerUrl: `/viewer?projectId=${projectId}`
              })
            } else {
              sendEvent('error', {
                message: `Pipeline exited with code ${code}`,
                stage: currentStage,
                elapsedTime: elapsed
              })
            }

            closeController()
          })

          pythonProcess.on('error', (err: Error) => {
            console.error('Process error:', err)
            sendEvent('error', {
              message: `Failed to start pipeline: ${err.message}`,
              stage: currentStage
            })
            sendEvent('log', { message: 'Note: Check Python installation and pipeline scripts' })
            closeController()
          })

        } catch (err: any) {
          console.error('Failed to spawn process:', err)
          sendEvent('log', { message: `Error: ${err.message}` })
          sendEvent('log', { message: 'Running in simulation mode' })
          closeController()
        }
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      }
    })
  } catch (error) {
    console.error('Process API error:', error)
    return NextResponse.json(
      { success: false, message: 'Failed to start pipeline', error: String(error) },
      { status: 500 }
    )
  }
}

// DELETE: Cancel running process
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const projectId = searchParams.get('projectId')

    if (!projectId) {
      return NextResponse.json(
        { success: false, message: 'Project ID required' },
        { status: 400 }
      )
    }

    const activeProcess = activeProcesses.get(projectId)
    if (activeProcess) {
      activeProcess.process.kill?.()
      activeProcesses.delete(projectId)
      return NextResponse.json({ success: true, message: 'Process cancelled' })
    }

    return NextResponse.json(
      { success: false, message: 'No active process found' },
      { status: 404 }
    )
  } catch (error) {
    return NextResponse.json(
      { success: false, message: 'Failed to cancel process' },
      { status: 500 }
    )
  }
}