import 'dart:convert';
import 'dart:math';
import 'dart:ui';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:path_provider/path_provider.dart';
import 'd_viewer_screen.dart'; // Ensure this path is correct

class ModelsScreen extends StatefulWidget {
  const ModelsScreen({super.key});

  @override
  State<ModelsScreen> createState() => _ModelsScreenState();
}

class _ModelsScreenState extends State<ModelsScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late List<Bubble> bubbles;
  final Random _random = Random();

  List<UserModel> userModels = [];
  bool isLoading = false;
  String? errorMsg;

  // No trailing slash here
  static const String baseUrl = 'http://10.113.79.190:8000/objfiles';

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 60),
    )..addListener(() {
      _updateBubbles();
      if (mounted) setState(() {});
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
    WidgetsBinding.instance.addPostFrameCallback((_) => _fetchModels());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _updateBubbles() {
    if (!mounted) return;
    final size = MediaQuery.of(context).size;
    final width = size.width;
    final height = size.height;

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

  Future<void> _fetchModels() async {
    if (!mounted) return;
    setState(() {
      isLoading = true;
      errorMsg = null;
    });

    try {
      final userId = FirebaseAuth.instance.currentUser?.uid;
      if (userId == null) throw Exception("User not logged in");

      final apiUrl = '$baseUrl/models/?user_id=$userId';
      final response = await http
          .get(Uri.parse(apiUrl))
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final List<dynamic> jsonData = json.decode(response.body);
        userModels =
            jsonData
                .map((e) {
                  String glbUrl = e['glb_url'] ?? '';
                  // Ensure the URL is fully qualified, join baseUrl and path correctly
                  if (glbUrl.isNotEmpty && glbUrl.startsWith('/')) {
                    glbUrl = '$baseUrl$glbUrl';
                  } else if (glbUrl.isNotEmpty && !glbUrl.startsWith('http')) {
                    glbUrl = '$baseUrl/$glbUrl';
                  }
                  return UserModel(
                    id: e['id']?.toString() ?? '',
                    title: e['title']?.toString() ?? 'Untitled',
                    thumbnailUrl: null,
                    glbUrl: glbUrl,
                  );
                })
                .where((model) => model.glbUrl.isNotEmpty)
                .toList();
      } else {
        errorMsg =
            'Failed to load models: ${response.statusCode}\nResponse: ${response.body}';
      }
    } catch (e) {
      errorMsg = 'Error loading models: $e';
    }

    if (mounted) setState(() => isLoading = false);
  }

  Future<void> _refreshModels() async => await _fetchModels();

  // Download model function
  Future<void> _downloadModel(UserModel model) async {
    try {
      _showSnackBar('Downloading ${model.title}...');

      // Download the file
      final response = await http.get(Uri.parse(model.glbUrl));
      if (response.statusCode == 200) {
        // Get the public Downloads directory
        Directory? directory;

        if (Platform.isAndroid) {
          // Use public Downloads folder
          directory = Directory('/storage/emulated/0/Download');
          // Also try alternative path
          if (!await directory.exists()) {
            directory = Directory('/storage/emulated/0/Downloads');
          }
        } else {
          // For iOS, use documents directory
          directory = await getApplicationDocumentsDirectory();
        }

        // Create safe filename (remove special characters)
        String safeTitle =
            model.title.replaceAll(RegExp(r'[^\w\s-]'), '').trim();
        if (safeTitle.isEmpty) safeTitle = 'model_${model.id}';
        String fileName = '$safeTitle.glb';

        final file = File('${directory.path}/$fileName');

        // Write the file
        await file.writeAsBytes(response.bodyBytes);

        _showSnackBar('Model saved to Downloads folder: $fileName');
      } else {
        _showSnackBar(
          'Failed to download model (Status: ${response.statusCode})',
        );
      }
    } catch (e) {
      _showSnackBar('Error downloading model: ${e.toString()}');
      print('Download error: $e'); // For debugging
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.deepPurple),
    );
  }

  // Show options menu
  void _showOptionsMenu(BuildContext context, UserModel model) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (BuildContext context) {
        return Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.9),
            borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 10),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              ListTile(
                leading: const Icon(Icons.download, color: Colors.deepPurple),
                title: const Text('Download'),
                onTap: () {
                  Navigator.pop(context);
                  _downloadModel(model);
                },
              ),
              ListTile(
                leading: const Icon(Icons.view_in_ar, color: Colors.deepPurple),
                title: const Text('View in 3D'),
                onTap: () {
                  Navigator.pop(context);
                  _openModelViewer(model);
                },
              ),
              const SizedBox(height: 20),
            ],
          ),
        );
      },
    );
  }

  Widget frostedCard({required Widget child}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.15),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white.withOpacity(0.2)),
          ),
          child: child,
        ),
      ),
    );
  }

  void _openModelViewer(UserModel model) {
    if (model.glbUrl.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Model URL is missing!')));
      return;
    }
    Navigator.of(context).push(
      MaterialPageRoute(
        builder:
            (context) => ModelViewerScreen(
              glbUrl: model.glbUrl,
              modelTitle: model.title,
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
        centerTitle: true,
        title: const Text(
          'My Models',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 24,
            color: Colors.white,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: _refreshModels,
          ),
        ],
      ),
      body: Stack(
        children: [
          // Gradient background
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
          CustomPaint(size: size, painter: BubblePainter(bubbles)),
          RefreshIndicator(
            onRefresh: _refreshModels,
            color: colorScheme.primary,
            backgroundColor: Colors.white.withOpacity(0.1),
            child:
                isLoading
                    ? const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          CircularProgressIndicator(
                            valueColor: AlwaysStoppedAnimation<Color>(
                              Colors.white,
                            ),
                          ),
                          SizedBox(height: 16),
                          Text(
                            'Loading models...',
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
                    )
                    : errorMsg != null
                    ? Center(
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(
                              Icons.error_outline,
                              color: Colors.redAccent,
                              size: 64,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              errorMsg!,
                              style: const TextStyle(
                                color: Colors.redAccent,
                                fontSize: 16,
                              ),
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: 20),
                            ElevatedButton(
                              onPressed: _refreshModels,
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.deepPurple,
                                foregroundColor: Colors.white,
                              ),
                              child: const Text('Retry'),
                            ),
                          ],
                        ),
                      ),
                    )
                    : userModels.isEmpty
                    ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.view_in_ar_outlined,
                            size: 64,
                            color: Colors.white.withOpacity(0.6),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'No models generated yet.',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.8),
                              fontSize: 18,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'Upload some images to generate 3D models!',
                            style: TextStyle(
                              color: Colors.white.withOpacity(0.6),
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    )
                    : SingleChildScrollView(
                      physics: const AlwaysScrollableScrollPhysics(),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 24,
                      ),
                      child: Center(
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 800),
                          child: GridView.builder(
                            shrinkWrap: true,
                            physics: const NeverScrollableScrollPhysics(),
                            itemCount: userModels.length,
                            gridDelegate:
                                const SliverGridDelegateWithFixedCrossAxisCount(
                                  crossAxisCount: 2,
                                  mainAxisSpacing: 20,
                                  crossAxisSpacing: 20,
                                  childAspectRatio: 0.8,
                                ),
                            itemBuilder: (context, index) {
                              final model = userModels[index];
                              return frostedCard(
                                child: Stack(
                                  children: [
                                    InkWell(
                                      borderRadius: BorderRadius.circular(16),
                                      onTap: () => _openModelViewer(model),
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.center,
                                        children: [
                                          Expanded(
                                            child: ClipRRect(
                                              borderRadius:
                                                  const BorderRadius.vertical(
                                                    top: Radius.circular(16),
                                                  ),
                                              child: Container(
                                                color: Colors.white.withOpacity(
                                                  0.1,
                                                ),
                                                child: const Icon(
                                                  Icons.view_in_ar_outlined,
                                                  size: 64,
                                                  color: Colors.white70,
                                                ),
                                              ),
                                            ),
                                          ),
                                          Padding(
                                            padding: const EdgeInsets.all(12.0),
                                            child: Column(
                                              children: [
                                                Text(
                                                  model.title,
                                                  style: TextStyle(
                                                    fontWeight: FontWeight.w600,
                                                    color: Colors.white
                                                        .withOpacity(0.9),
                                                    fontSize: 16,
                                                  ),
                                                  textAlign: TextAlign.center,
                                                  maxLines: 2,
                                                  overflow:
                                                      TextOverflow.ellipsis,
                                                ),
                                                const SizedBox(height: 4),
                                                Text(
                                                  'Tap to view in 3D',
                                                  style: TextStyle(
                                                    color: Colors.white
                                                        .withOpacity(0.6),
                                                    fontSize: 12,
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    // Three dots menu button
                                    Positioned(
                                      top: 8,
                                      right: 8,
                                      child: GestureDetector(
                                        onTap:
                                            () => _showOptionsMenu(
                                              context,
                                              model,
                                            ),
                                        child: Container(
                                          padding: const EdgeInsets.all(4),
                                          decoration: BoxDecoration(
                                            color: Colors.black.withOpacity(
                                              0.3,
                                            ),
                                            borderRadius: BorderRadius.circular(
                                              12,
                                            ),
                                          ),
                                          child: const Icon(
                                            Icons.more_vert,
                                            color: Colors.white,
                                            size: 20,
                                          ),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                    ),
          ),
        ],
      ),
    );
  }
}

class UserModel {
  final String id;
  final String title;
  final String? thumbnailUrl;
  final String glbUrl;

  UserModel({
    required this.id,
    required this.title,
    this.thumbnailUrl,
    required this.glbUrl,
  });
}

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
