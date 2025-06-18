import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class MyProfileScreen extends StatefulWidget {
  const MyProfileScreen({super.key});

  @override
  State<MyProfileScreen> createState() => _MyProfileScreenState();
}

class _MyProfileScreenState extends State<MyProfileScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late List<Bubble> bubbles;
  final Random _random = Random();

  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  Map<String, dynamic>? userData;
  bool isLoading = true;
  String? error;

  @override
  void initState() {
    super.initState();
    _initializeAnimation();
    _fetchUserData();
  }

  void _initializeAnimation() {
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

  Future<void> _fetchUserData() async {
    try {
      User? user = _auth.currentUser;
      if (user != null) {
        DocumentSnapshot doc =
            await _firestore.collection('users').doc(user.uid).get();

        if (doc.exists) {
          setState(() {
            userData = doc.data() as Map<String, dynamic>?;
            isLoading = false;
          });
        } else {
          setState(() {
            error = 'User data not found';
            isLoading = false;
          });
        }
      } else {
        setState(() {
          error = 'No user logged in';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        error = 'Failed to load profile data: $e';
        isLoading = false;
      });
    }
  }

  Widget frostedCard({required Widget child}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(20),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.15),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(0.2)),
          ),
          padding: const EdgeInsets.all(20),
          child: child,
        ),
      ),
    );
  }

  Widget _buildProfileAvatar() {
    String initials = '';
    if (userData != null) {
      String firstName = userData!['firstName'] ?? '';
      String lastName = userData!['lastName'] ?? '';
      if (firstName.isNotEmpty) initials += firstName[0].toUpperCase();
      if (lastName.isNotEmpty) initials += lastName[0].toUpperCase();
    }

    return Container(
      width: 120,
      height: 120,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        gradient: LinearGradient(
          colors: [Colors.pink.shade300, Colors.purple.shade600],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.purple.withOpacity(0.3),
            blurRadius: 20,
            spreadRadius: 5,
          ),
        ],
      ),
      child: Center(
        child: Text(
          initials,
          style: const TextStyle(
            fontSize: 36,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
        ),
      ),
    );
  }

  Widget _buildInfoCard(
    IconData icon,
    String title,
    String value,
    Color iconColor,
  ) {
    return frostedCard(
      child: Row(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: iconColor.withOpacity(0.2),
              borderRadius: BorderRadius.circular(15),
            ),
            child: Icon(icon, color: iconColor, size: 28),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.white.withOpacity(0.8),
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: const TextStyle(
                    fontSize: 16,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _formatDate(dynamic timestamp) {
    if (timestamp == null) return 'Not specified';

    try {
      if (timestamp is Timestamp) {
        DateTime date = timestamp.toDate();
        return '${date.day}/${date.month}/${date.year}';
      } else if (timestamp is String) {
        return timestamp;
      }
      return 'Invalid date';
    } catch (e) {
      return 'Invalid date';
    }
  }

  String _formatJoinDate(dynamic timestamp) {
    if (timestamp == null) return 'Unknown';

    try {
      if (timestamp is Timestamp) {
        DateTime date = timestamp.toDate();
        List<String> months = [
          'Jan',
          'Feb',
          'Mar',
          'Apr',
          'May',
          'Jun',
          'Jul',
          'Aug',
          'Sep',
          'Oct',
          'Nov',
          'Dec',
        ];
        return '${months[date.month - 1]} ${date.year}';
      }
      return 'Unknown';
    } catch (e) {
      return 'Unknown';
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: colorScheme.primary,
        title: const Text(
          'My Profile',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        elevation: 6,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
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

          // Animated bubbles
          CustomPaint(size: size, painter: BubblePainter(bubbles)),

          // Profile content
          if (isLoading)
            const Center(
              child: CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              ),
            )
          else if (error != null)
            Center(
              child: frostedCard(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.error_outline,
                      color: Colors.red,
                      size: 48,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      error!,
                      style: const TextStyle(color: Colors.white, fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton(
                      onPressed: () {
                        setState(() {
                          isLoading = true;
                          error = null;
                        });
                        _fetchUserData();
                      },
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              ),
            )
          else
            SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
              child: Column(
                children: [
                  // Profile Header
                  frostedCard(
                    child: Column(
                      children: [
                        _buildProfileAvatar(),
                        const SizedBox(height: 16),
                        Text(
                          userData?['fullName'] ?? 'Unknown User',
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.green.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: Colors.green.withOpacity(0.3),
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Container(
                                width: 8,
                                height: 8,
                                decoration: const BoxDecoration(
                                  color: Colors.green,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 8),
                              const Text(
                                'Active',
                                style: TextStyle(
                                  color: Colors.green,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Member since ${_formatJoinDate(userData?['createdAt'])}',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.white.withOpacity(0.7),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 20),

                  // Personal Information
                  _buildInfoCard(
                    Icons.person,
                    'Full Name',
                    userData?['fullName'] ?? 'Not specified',
                    Colors.blue,
                  ),

                  const SizedBox(height: 16),

                  _buildInfoCard(
                    Icons.email,
                    'Email Address',
                    userData?['email'] ?? 'Not specified',
                    Colors.orange,
                  ),

                  const SizedBox(height: 16),

                  _buildInfoCard(
                    Icons.wc,
                    'Gender',
                    userData?['gender'] ?? 'Not specified',
                    Colors.pink,
                  ),

                  const SizedBox(height: 16),

                  _buildInfoCard(
                    Icons.cake,
                    'Date of Birth',
                    _formatDate(userData?['dateOfBirth']),
                    Colors.purple,
                  ),

                  const SizedBox(height: 16),

                  _buildInfoCard(
                    Icons.fingerprint,
                    'User ID',
                    userData?['uid']?.substring(0, 8) ?? 'Unknown',
                    Colors.teal,
                  ),

                  const SizedBox(height: 30),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

// Bubble animation classes (same as your settings screen)
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
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
