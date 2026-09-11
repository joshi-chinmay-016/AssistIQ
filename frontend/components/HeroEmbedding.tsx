"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { EvidenceCase, LatencyInfo } from "@/lib/types";

interface HeroEmbeddingProps {
  isAnalyzing: boolean;
  evidence: EvidenceCase[] | null;
  latency: LatencyInfo | null;
  predictedIntent?: string | null;
}

export default function HeroEmbedding({
  isAnalyzing,
  evidence,
  latency,
  predictedIntent,
}: HeroEmbeddingProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Keep references to animate dynamically without tearing down scene
  const queryMeshRef = useRef<THREE.Mesh | null>(null);
  const connectionLinesRef = useRef<THREE.LineSegments | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    let animationFrameId: number;
    const prefersReducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    // 1. Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.fog = new THREE.FogExp2(0x090a0d, 0.012);

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 220;

    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    camera.position.set(0, 15, 65);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // 2. Generate Semantic Clustered Embedding Points
    // Simulates the 40,794 Spotify support embeddings distributed across intent clusters
    const clusterCenters = [
      { x: -18, y: 5, z: -10, color: 0x00d4c8 }, // Playback (Cyan)
      { x: 16, y: -6, z: 8, color: 0xf59e0b },  // Billing (Amber)
      { x: 0, y: 14, z: -14, color: 0x8b5cf6 },  // Account / Login (Purple)
      { x: -12, y: -10, z: 12, color: 0x10b981 },// Offline / Playlist (Emerald)
      { x: 18, y: 10, z: -6, color: 0x3b82f6 },  // Content / Meta (Blue)
    ];

    const totalParticles = 650;
    const positions = new Float32Array(totalParticles * 3);
    const colors = new Float32Array(totalParticles * 3);

    let idx = 0;
    for (let i = 0; i < totalParticles; i++) {
      const cluster = clusterCenters[i % clusterCenters.length];
      const spread = 8.5;

      const px = cluster.x + (Math.random() - 0.5) * spread * 2;
      const py = cluster.y + (Math.random() - 0.5) * spread * 1.5;
      const pz = cluster.z + (Math.random() - 0.5) * spread * 2;

      positions[idx] = px;
      positions[idx + 1] = py;
      positions[idx + 2] = pz;

      const c = new THREE.Color(cluster.color);
      // Add slight jitter to colors for organic embedding scatter
      c.offsetHSL(0, 0, (Math.random() - 0.5) * 0.15);
      colors[idx] = c.r;
      colors[idx + 1] = c.g;
      colors[idx + 2] = c.b;

      idx += 3;
    }

    const pointsGeometry = new THREE.BufferGeometry();
    pointsGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(positions, 3)
    );
    pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    // Particle texture
    const canvasTexture = document.createElement("canvas");
    canvasTexture.width = 16;
    canvasTexture.height = 16;
    const ctx = canvasTexture.getContext("2d");
    if (ctx) {
      const gradient = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
      gradient.addColorStop(0, "rgba(255,255,255,1)");
      gradient.addColorStop(0.4, "rgba(255,255,255,0.7)");
      gradient.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 16, 16);
    }
    const texture = new THREE.CanvasTexture(canvasTexture);

    const pointsMaterial = new THREE.PointsMaterial({
      size: 2.2,
      vertexColors: true,
      map: texture,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
    scene.add(pointCloud);

    // 3. Coordinate Grid rings for radar visual
    const ringGeo = new THREE.RingGeometry(24, 24.3, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x1f222e,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.4,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    scene.add(ring);

    // 4. Query point (Dynamic node)
    const queryGeo = new THREE.SphereGeometry(1.2, 16, 16);
    const queryMat = new THREE.MeshBasicMaterial({
      color: 0x00d4c8,
      wireframe: false,
    });
    const queryMesh = new THREE.Mesh(queryGeo, queryMat);
    queryMesh.position.set(0, 0, 0);
    queryMesh.visible = false;
    scene.add(queryMesh);
    queryMeshRef.current = queryMesh;

    // 5. Animation Loop
    const clock = new THREE.Clock();

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Gentle rotational drift
      if (!prefersReducedMotion) {
        pointCloud.rotation.y = elapsedTime * 0.05;
        pointCloud.rotation.x = Math.sin(elapsedTime * 0.03) * 0.05;
        ring.rotation.z = elapsedTime * 0.02;
      }

      renderer.render(scene, camera);
    };

    animate();

    // 6. Resize Observer
    const handleResize = () => {
      if (!container) return;
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(container);

    // Cleanup on unmount
    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      pointsGeometry.dispose();
      pointsMaterial.dispose();
      ringGeo.dispose();
      ringMat.dispose();
      queryGeo.dispose();
      queryMat.dispose();
      texture.dispose();
      renderer.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Update query point and connection lines when analyzing or when evidence changes
  useEffect(() => {
    const scene = sceneRef.current;
    const queryMesh = queryMeshRef.current;
    if (!scene || !queryMesh) return;

    // Remove existing connection lines
    if (connectionLinesRef.current) {
      scene.remove(connectionLinesRef.current);
      connectionLinesRef.current.geometry.dispose();
      (connectionLinesRef.current.material as THREE.Material).dispose();
      connectionLinesRef.current = null;
    }

    if (isAnalyzing) {
      // Reveal animated query vector entering the space
      queryMesh.visible = true;
      queryMesh.position.set(0, 0, 0);
      (queryMesh.material as THREE.MeshBasicMaterial).color.setHex(0xf59e0b);
    } else if (evidence && evidence.length > 0) {
      queryMesh.visible = true;
      (queryMesh.material as THREE.MeshBasicMaterial).color.setHex(0x00d4c8);

      // Determine center based on intent or neutral
      const queryPos = new THREE.Vector3(0, 0, 0);
      queryMesh.position.copy(queryPos);

      // Construct dynamic nearest-neighbor line segments to Top-K evidence
      const linePositions: number[] = [];
      const k = Math.min(evidence.length, 8);

      for (let i = 0; i < k; i++) {
        // Distribute neighbor coordinates around query node proportional to similarity
        const angle = (i / k) * Math.PI * 2;
        const sim = evidence[i].similarity || 0.75;
        // Higher similarity = closer neighbor
        const distance = 14 + (1 - sim) * 30;

        const nx = queryPos.x + Math.cos(angle) * distance;
        const ny = queryPos.y + (i % 2 === 0 ? 3 : -3);
        const nz = queryPos.z + Math.sin(angle) * distance;

        // Line from query to neighbor
        linePositions.push(queryPos.x, queryPos.y, queryPos.z);
        linePositions.push(nx, ny, nz);
      }

      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(linePositions, 3)
      );

      const lineMat = new THREE.LineBasicMaterial({
        color: 0x00d4c8,
        transparent: true,
        opacity: 0.65,
        blending: THREE.AdditiveBlending,
      });

      const lineSegments = new THREE.LineSegments(lineGeo, lineMat);
      scene.add(lineSegments);
      connectionLinesRef.current = lineSegments;
    } else {
      queryMesh.visible = false;
    }
  }, [isAnalyzing, evidence, predictedIntent]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-[180px] sm:h-[210px] bg-[#0b0c10] border-b border-[#1f222e] overflow-hidden select-none"
    >
      {/* Three.js Canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 block w-full h-full" />

      {/* Top Left Status Badge */}
      <div className="absolute top-3 left-4 pointer-events-none flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#00d4c8] animate-pulse" />
          <span className="text-[10px] font-mono uppercase tracking-widest text-[#00d4c8]">
            Embedding Space
          </span>
        </div>
        <p className="text-[11px] font-mono text-[#858a98] tracking-tight">
          Retrieving from 40,794 historical SpotifyCares conversations
        </p>
      </div>

      {/* Top Right Cluster Index Indicator */}
      <div className="absolute top-3 right-4 pointer-events-none hidden sm:flex items-center gap-3 text-[10px] font-mono text-[#555967]">
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00d4c8]" />
          <span>Playback</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#f59e0b]" />
          <span>Billing</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#8b5cf6]" />
          <span>Account</span>
        </div>
      </div>

      {/* Dynamic Processing Overlay */}
      {isAnalyzing && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#090a0d]/50 backdrop-blur-[1px] pointer-events-none">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-[#111217]/90 border border-[#f59e0b]/40 shadow-[0_0_15px_rgba(245,158,11,0.2)]">
            <span className="w-2 h-2 rounded-full bg-[#f59e0b] animate-ping" />
            <span className="text-xs font-mono tracking-wider text-[#f59e0b] uppercase">
              Mapping Query Vector to FAISS Space...
            </span>
          </div>
        </div>
      )}

      {/* Bottom Telemetry Bar */}
      {latency && !isAnalyzing && (
        <div className="absolute bottom-2 left-4 right-4 flex items-center justify-between px-2.5 py-1 rounded bg-[#111217]/85 border border-[#1f222e] text-[10px] font-mono text-[#858a98] pointer-events-none">
          <span className="text-[#eae8e3]">Telemetry:</span>
          <div className="flex items-center gap-3 overflow-x-auto">
            <span>
              Intent: <strong className="text-[#00d4c8]">{latency.intent_ms.toFixed(1)}ms</strong>
            </span>
            <span>
              FAISS: <strong className="text-[#00d4c8]">{latency.retrieval_ms.toFixed(1)}ms</strong>
            </span>
            <span>
              Gemini: <strong className="text-[#00d4c8]">{latency.generation_ms.toFixed(1)}ms</strong>
            </span>
            <span>
              Policy: <strong className="text-[#00d4c8]">{latency.escalation_ms.toFixed(1)}ms</strong>
            </span>
            <span className="text-[#eae8e3]">
              Total: <strong>{latency.total_ms.toFixed(1)}ms</strong>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
