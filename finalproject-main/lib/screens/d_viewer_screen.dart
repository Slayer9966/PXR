import 'package:flutter/material.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';

class ModelViewerScreen extends StatefulWidget {
  final String glbUrl;
  final String? modelTitle;

  const ModelViewerScreen({Key? key, required this.glbUrl, this.modelTitle})
    : super(key: key);

  @override
  State<ModelViewerScreen> createState() => _ModelViewerScreenState();
}

class _ModelViewerScreenState extends State<ModelViewerScreen> {
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    // Simulate loading for 2 seconds for UX
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => isLoading = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.modelTitle ?? '3D Model Viewer'),
        backgroundColor: Colors.deepPurple,
        centerTitle: true,
        elevation: 4,
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Stack(
        children: [
          if (!isLoading)
            SizedBox.expand(
              child: ModelViewer(
                src: widget.glbUrl,
                alt: "3D Model",
                autoRotate: true,
                cameraControls: true,
                disableZoom: false,
                disablePan: false,
                ar: false,
                // src: widget.glbUrl,
                // alt: widget.modelTitle ?? "3D Model",
                // ar: false,
                // autoRotate: true,
                // cameraControls: true,
                backgroundColor: const Color.fromARGB(255, 12, 12, 12),
                // disableZoom: false,
                // shadowIntensity: 0.3,
                // shadowSoftness: 0.5,
                // disablePan: false,

                // cameraOrbit: "0deg 75deg 105%",
                // minCameraOrbit: "auto auto auto",
                // maxCameraOrbit: "auto auto auto",
              ),
            ),
          if (isLoading)
            Container(
              color: Colors.black.withOpacity(0.3),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'Loading 3D Model...',
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'This may take a moment',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
