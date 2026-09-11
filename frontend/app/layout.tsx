import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const ibmPlexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-ibm-sans",
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-ibm-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "AssistIQ — SpotifyCares AI Support Console",
  description: "Deterministic AI Support Console for SpotifyCares with Intent Classification, FAISS Retrieval, Grounded Reply Generation, and Escalation Routing.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${ibmPlexSans.variable} ${ibmPlexMono.variable} h-full dark`}
    >
      <body className="min-h-full flex flex-col bg-[#090a0d] text-[#eae8e3] font-sans antialiased selection:bg-[#00d4c8]/20 selection:text-[#00d4c8]">
        {children}
      </body>
    </html>
  );
}
