import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:http_parser/http_parser.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen>
    with SingleTickerProviderStateMixin {
  List<PlatformFile> _selectedFiles = [];
  late AnimationController _controller;
  late List<Bubble> bubbles;
  final Random _random = Random();
  Future<void> _uploadFiles() async {
    const String uploadUrl =
        'http://10.113.79.190:8000/objfiles/upload-images/';
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Not logged in!')));
      return;
    }
    String userId = user.uid; // This is the Firebase UID

    var request = http.MultipartRequest('POST', Uri.parse(uploadUrl))
      ..fields['user_id'] = userId; // Send UID (string)

    for (var file in _selectedFiles) {
      if (file.path == null) continue;

      request.files.add(
        await http.MultipartFile.fromPath(
          'files',
          file.path!,
          filename: path.basename(file.path!),
          contentType: MediaType(
            'image',
            path.extension(file.path!).replaceFirst('.', ''),
          ),
        ),
      );
    }

    var response = await request.send();

    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      debugPrint('Upload successful: $responseBody');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Upload successful')));
    } else {
      debugPrint('Upload failed with status: ${response.statusCode}');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Upload failed')));
    }
  }

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 60),
    )..addListener(() {
      _updateBubbles();
      setState(() {});
    });

    bubbles = List.generate(7, (index) {
      final radius = _random.nextDouble() * 40 + 20;
      return Bubble(
        position: Offset(
          _random.nextDouble() * 400,
          _random.nextDouble() * 800,
        ),
        radius: radius,
        speed: _random.nextDouble() * 0.5 + 0.1,
        direction: Offset(
          (_random.nextDouble() - 0.5) * 0.4,
          (_random.nextDouble() - 0.5) * 0.4,
        ),
      );
    });

    _controller.repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _updateBubbles() {
    final width = MediaQuery.of(context).size.width;
    final height = MediaQuery.of(context).size.height;

    for (var bubble in bubbles) {
      bubble.position += bubble.direction * bubble.speed;

      if (bubble.position.dx < 0 || bubble.position.dx > width) {
        bubble.direction = Offset(-bubble.direction.dx, bubble.direction.dy);
      }
      if (bubble.position.dy < 0 || bubble.position.dy > height) {
        bubble.direction = Offset(bubble.direction.dx, -bubble.direction.dy);
      }
    }
  }

  Future<void> _pickFiles() async {
    final result = await FilePicker.platform.pickFiles(
      allowMultiple: true,
      type: FileType.any,
    );

    if (result != null) {
      setState(() {
        _selectedFiles = result.files;
      });
    }
  }

  void _clearFiles() {
    setState(() {
      _selectedFiles.clear();
    });
  }

  Widget frostedCard({required Widget child}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.2),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white.withOpacity(0.3)),
          ),
          child: child,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final size = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: colorScheme.primary,
        elevation: 6,
        title: const Text(
          'Upload Files',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 24,
            color: Colors.white,
          ),
        ),
        centerTitle: true,
      ),
      body: Stack(
        children: [
          // Gradient background - exactly like HomeScreen
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Colors.deepPurple.shade900,
                  Colors.purple.shade600,
                  Colors.pink.shade400,
                ],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),

          // Animated bubbles behind content - exactly like HomeScreen
          CustomPaint(size: size, painter: BubblePainter(bubbles)),

          // Content with padding
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 24, 20, 20),
            child: Column(
              children: [
                frostedCard(
                  child: InkWell(
                    onTap: _pickFiles,
                    borderRadius: BorderRadius.circular(16),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(vertical: 40),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.cloud_upload_outlined,
                            size: 64,
                            color: Colors.white,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Tap to select files\n(Supports multiple)',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w600,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 30),
                Expanded(
                  child:
                      _selectedFiles.isEmpty
                          ? Center(
                            child: Text(
                              'No files selected',
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.7),
                                fontSize: 16,
                              ),
                            ),
                          )
                          : GridView.builder(
                            itemCount: _selectedFiles.length,
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                                  crossAxisCount: 3,
                                  crossAxisSpacing: 16,
                                  mainAxisSpacing: 16,
                                ),
                            itemBuilder: (context, index) {
                              final file = _selectedFiles[index];
                              final isImage =
                                  file.extension != null &&
                                  [
                                    'jpg',
                                    'jpeg',
                                    'png',
                                    'gif',
                                    'bmp',
                                  ].contains(file.extension!.toLowerCase());

                              return frostedCard(
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(12),
                                  child:
                                      isImage && file.path != null
                                          ? Image.file(
                                            File(file.path!),
                                            fit: BoxFit.cover,
                                          )
                                          : Center(
                                            child: Icon(
                                              Icons.insert_drive_file_outlined,
                                              color: Colors.white,
                                              size: 40,
                                            ),
                                          ),
                                ),
                              );
                            },
                          ),
                ),
                if (_selectedFiles.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 16),
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: colorScheme.error,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 32,
                          vertical: 14,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      icon: const Icon(Icons.clear),
                      label: const Text('Clear All'),
                      onPressed: _clearFiles,
                    ),
                  ),
                if (_selectedFiles.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 12),
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 32,
                          vertical: 14,
                        ),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      icon: const Icon(Icons.upload_file),
                      label: const Text('Upload'),
                      onPressed: _uploadFiles,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Bubble class and painter copied from HomeScreen

class Bubble {
  Offset position;
  final double radius;
  final double speed;
  Offset direction;

  Bubble({
    required this.position,
    required this.radius,
    required this.speed,
    required this.direction,
  });
}

class BubblePainter extends CustomPainter {
  final List<Bubble> bubbles;

  BubblePainter(this.bubbles);

  @override
  void paint(Canvas canvas, Size size) {
    final gradient = RadialGradient(
      colors: [
        Colors.pink.shade300.withOpacity(0.3),
        Colors.purple.shade700.withOpacity(0.1),
      ],
    );

    final paint = Paint()..style = PaintingStyle.fill;

    for (final bubble in bubbles) {
      final rect = Rect.fromCircle(
        center: bubble.position,
        radius: bubble.radius,
      );
      paint.shader = gradient.createShader(rect);
      canvas.drawCircle(bubble.position, bubble.radius, paint);
    }
  }

  @override
  bool shouldRepaint(covariant BubblePainter oldDelegate) => true;
}
