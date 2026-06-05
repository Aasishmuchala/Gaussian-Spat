import { NextRequest, NextResponse } from 'next/server'
import { writeFile, mkdir, readdir } from 'fs/promises'
import { existsSync } from 'fs'
import path from 'path'
import { randomUUID } from 'crypto'

const UPLOAD_DIR = path.join(process.cwd(), 'uploads')

export async function POST(request: NextRequest) {
  try {
    // Ensure upload directory exists
    if (!existsSync(UPLOAD_DIR)) {
      await mkdir(UPLOAD_DIR, { recursive: true })
    }

    const formData = await request.formData()
    const files = formData.getAll('files') as File[]

    if (!files || files.length === 0) {
      return NextResponse.json(
        { success: false, message: 'No files uploaded' },
        { status: 400 }
      )
    }

    // Create project directory
    const projectId = randomUUID()
    const projectDir = path.join(UPLOAD_DIR, projectId)
    await mkdir(projectDir, { recursive: true })

    // Also create videos subdirectory
    const videosDir = path.join(projectDir, 'videos')
    await mkdir(videosDir, { recursive: true })

    const uploadedFiles: { name: string; path: string; size: number }[] = []
    const videoPaths: string[] = []

    // Save each file
    for (const file of files) {
      const buffer = Buffer.from(await file.arrayBuffer())
      const filename = file.name.replace(/[^a-zA-Z0-9.-]/g, '_')
      const filepath = path.join(videosDir, filename)

      await writeFile(filepath, buffer)
      uploadedFiles.push({
        name: filename,
        path: filepath,
        size: buffer.length
      })
      videoPaths.push(filepath)

      console.log(`Uploaded: ${filename} (${(buffer.length / 1024 / 1024).toFixed(2)} MB)`)
    }

    return NextResponse.json({
      success: true,
      projectId,
      files: uploadedFiles,
      videoPaths, // Return the actual file paths for the pipeline
      videosDir,
      message: `Successfully uploaded ${files.length} file(s)`
    })
  } catch (error) {
    console.error('Upload error:', error)
    return NextResponse.json(
      { success: false, message: 'Upload failed', error: String(error) },
      { status: 500 }
    )
  }
}

// GET: List recent uploads
export async function GET() {
  try {
    if (!existsSync(UPLOAD_DIR)) {
      return NextResponse.json({ success: true, projects: [] })
    }

    const entries = await readdir(UPLOAD_DIR, { withFileTypes: true })
    const projects = entries
      .filter(e => e.isDirectory())
      .map(e => ({ id: e.name, path: path.join(UPLOAD_DIR, e.name) }))

    return NextResponse.json({ success: true, projects })
  } catch (error) {
    return NextResponse.json(
      { success: false, message: 'Failed to list uploads' },
      { status: 500 }
    )
  }
}