import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vectorless Video Chatbot",
  description: "Chat with your videos without RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="min-h-screen flex flex-col">
          <main className="flex-1 bg-gray-50">
            {children}
          </main>
          <footer className="bg-gray-100 border-t border-gray-200 py-6">
            <div className="container mx-auto px-4 text-center">
              <p className="text-gray-600 text-sm">
                Authored by <a href="https://splashing23.github.io/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Dev Gupta</a>
              </p>
            </div>
          </footer>
        </div>
        <Analytics />
      </body>
    </html>
  );
}
