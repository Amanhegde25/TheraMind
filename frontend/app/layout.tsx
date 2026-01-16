import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI Therapist - Mental Wellness Support',
  description: 'Your compassionate AI companion for mental health support and emotional well-being',
  keywords: ['mental health', 'AI therapist', 'emotional support', 'wellness'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}