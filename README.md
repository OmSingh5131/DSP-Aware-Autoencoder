
# 🛰️ DSP-Aware Convolutional Autoencoder for Satellite Image Compression in 5G Network Slicing

## 💡 Project Overview

This project presents a novel, end-to-end deep learning system for high-fidelity satellite image compression and ultra-reliable transmission. This system was implemented after reading multiple research papers on deep learning, image compression, and 5G telecommunications. It addresses the fundamental limitation of traditional codecs like JPEG—the aggressive suppression of high-frequency components critical for scientific analysis—by introducing a **DSP-Aware Convolutional Autoencoder**.

Furthermore, it integrates this AI compression engine with **5G Network Slicing** to create a **Semantic-Aware Transmission** pipeline, ensuring critical decoding parameters are never lost, even in poor channel conditions.

### Key Innovations

* **DSP-Aware Compression:** Replaces fixed mathematical transforms (Discrete Cosine Transform) with a deep learning model that learns non-linear transforms tailored to preserve fine, high-frequency satellite details.
* **Hyperprior Architecture:** Utilizes a sophisticated two-stream architecture to decompose the image into a high-volume **Bulk Latent Vector ($y$)** and a mission-critical, tiny **Entropy Key ($z$)**.
* **Custom Loss Function:** Employs a composite DSP-Aware Loss Function, including a **Perceptual Loss ($P$)** based on VGG-19 features, to enforce structural integrity and prevent blurring/blocking artifacts common in JPEG.
* **5G Network Slicing for Reliability:** Simulates a 5G downlink that routes the Bulk data ($y$) through a high-throughput **eMBB slice (64-QAM)** and the critical Key ($z$) through an ultra-reliable **URLLC slice (QPSK)**.

---

