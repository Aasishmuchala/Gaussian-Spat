/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: ['three', '@react-three/fiber', '@react-three/drei'],
  experimental: {
    serverActions: {
      bodySizeLimit: '10gb',
    },
  },
}

module.exports = nextConfig