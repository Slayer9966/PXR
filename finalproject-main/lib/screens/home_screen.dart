import 'dart:math';
import 'package:flutter/material.dart';
import '/widgets/Navigation.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late List<Bubble> bubbles;
  final Random _random = Random();

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

  Widget featureCard(
    BuildContext context,
    IconData icon,
    String label,
    VoidCallback onTap,
  ) {
    final colorScheme = Theme.of(context).colorScheme;

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        width: 140,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: colorScheme.primaryContainer.withOpacity(0.85),
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: colorScheme.primary.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 48, color: colorScheme.primary),
            const SizedBox(height: 12),
            Text(
              label,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: colorScheme.onPrimaryContainer,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  void _navigateToPage(int index) {
    // This assumes you have the same order in your navigation as in the bottom nav:
    // 0: Home, 1: Upload, 2: Models, 3: Settings
    final mainNavState = context.findAncestorStateOfType<MainNavigationState>();
    if (mainNavState != null) {
      mainNavState.onItemTapped(index);
    }
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
          'PXR',
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

          // Animated bubbles behind content
          CustomPaint(size: size, painter: BubblePainter(bubbles)),

          // Centered content
          Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              child: ConstrainedBox(
                constraints: BoxConstraints(maxWidth: 600),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Text(
                      'Welcome to PXR!',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.white.withOpacity(0.8),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Transform 2D images into immersive 3D and AR experiences.',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.white.withOpacity(0.8),
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 32),
                    Wrap(
                      alignment: WrapAlignment.center,
                      spacing: 20,
                      runSpacing: 20,
                      children: [
                        featureCard(
                          context,
                          Icons.upload_file_outlined,
                          'Upload',
                          () => _navigateToPage(1),
                        ),
                        featureCard(
                          context,
                          Icons.grid_view_outlined,
                          'Models',
                          () => _navigateToPage(2),
                        ),
                        featureCard(
                          context,
                          Icons.settings_outlined,
                          'Settings',
                          () => _navigateToPage(3),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
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
