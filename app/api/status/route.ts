import { NextRequest, NextResponse } from 'next/server'

// Store status for each project
const projectStatuses = new Map<string, {
  stage: string
  progress: number
  message: string
  elapsedTime?: number
  estimatedTimeRemaining?: number
  error?: string
}>()

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const projectId = searchParams.get('projectId')

  if (!projectId) {
    return NextResponse.json(
      { success: false, message: 'Project ID required' },
      { status: 400 }
    )
  }

  const status = projectStatuses.get(projectId)

  if (!status) {
    return NextResponse.json({
      success: true,
      status: {
        stage: 'idle',
        progress: 0,
        message: 'No active process'
      }
    })
  }

  return NextResponse.json({ success: true, status })
}

// POST: Update status
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { projectId, ...statusUpdate } = body

    if (!projectId) {
      return NextResponse.json(
        { success: false, message: 'Project ID required' },
        { status: 400 }
      )
    }

    const currentStatus = projectStatuses.get(projectId) || {
      stage: 'idle',
      progress: 0,
      message: ''
    }

    projectStatuses.set(projectId, { ...currentStatus, ...statusUpdate })

    return NextResponse.json({ success: true, status: projectStatuses.get(projectId) })
  } catch (error) {
    return NextResponse.json(
      { success: false, message: 'Failed to update status' },
      { status: 500 }
    )
  }
}

// SSE stream for real-time updates
export async function put(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const projectId = searchParams.get('projectId')

  if (!projectId) {
    return NextResponse.json(
      { success: false, message: 'Project ID required' },
      { status: 400 }
    )
  }

  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    start(controller) {
      // Send initial status
      const status = projectStatuses.get(projectId) || {
        stage: 'idle',
        progress: 0,
        message: 'Waiting for updates...'
      }
      controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'status', status })}\n\n`))

      // Poll for updates every second (in production, use WebSocket or proper SSE)
      const interval = setInterval(() => {
        const currentStatus = projectStatuses.get(projectId)
        if (currentStatus) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: 'status', status: currentStatus })}\n\n`))
        }

        // Close if process is complete or errored
        if (currentStatus?.stage === 'complete' || currentStatus?.stage === 'error') {
          clearInterval(interval)
          controller.close()
        }
      }, 1000)

      // Cleanup
      return () => clearInterval(interval)
    }
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    }
  })
}