import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: '4D Gaussian Splatting Studio',
  description: 'Transform multi-view video into dynamic 4D Gaussian splatting reconstructions',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <header className="border-b">
            <div className="container mx-auto px-4 py-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <span className="text-white font-bold text-lg">4D</span>
                </div>
                <div>
                  <h1 className="text-xl font-bold">4DGS Studio</h1>
                  <p className="text-xs text-muted-foreground">Gaussian Splatting Studio</p>
                </div>
              </div>
              <nav className="flex items-center gap-4">
                <a href="/" className="text-sm hover:text-primary transition-colors">
                  Home
                </a>
                <a href="/viewer" className="text-sm hover:text-primary transition-colors">
                  Viewer
                </a>
              </nav>
            </div>
          </header>
          <main className="flex-1">
            {children}
          </main>
          <footer className="border-t py-6">
            <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
              Powered by FreeTimeGS • Built with Next.js
            </div>
          </footer>
        </div>
      </body>
    </html>
  )
}