import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class EditProfilePage extends StatefulWidget {
  const EditProfilePage({super.key});

  @override
  State<EditProfilePage> createState() => _EditProfilePageState();
}

class _EditProfilePageState extends State<EditProfilePage>
    with TickerProviderStateMixin {
  final TextEditingController _firstNameController = TextEditingController();
  final TextEditingController _lastNameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _dobController = TextEditingController();
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  AnimationController? _animationController;
  Animation<double>? _animation;
  String _selectedGender = '';
  DateTime? _selectedDate;
  bool _isLoading = false;
  bool _isLoadingData = true;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 10),
      vsync: this,
    )..repeat();
    _animation = Tween<double>(begin: 0, end: 1).animate(_animationController!);
    _loadUserData();
  }

  @override
  void dispose() {
    _animationController?.dispose();
    _firstNameController.dispose();
    _lastNameController.dispose();
    _emailController.dispose();
    _dobController.dispose();
    super.dispose();
  }

  Future<void> _loadUserData() async {
    try {
      User? currentUser = _auth.currentUser;
      if (currentUser != null) {
        DocumentSnapshot userDoc =
            await _firestore.collection('users').doc(currentUser.uid).get();

        if (userDoc.exists) {
          Map<String, dynamic> userData =
              userDoc.data() as Map<String, dynamic>;

          setState(() {
            _firstNameController.text = userData['firstName'] ?? '';
            _lastNameController.text = userData['lastName'] ?? '';
            _emailController.text = userData['email'] ?? '';
            _selectedGender = userData['gender'] ?? '';
            _dobController.text = userData['dateOfBirth'] ?? '';

            // Parse the date if it exists
            if (userData['dateOfBirth'] != null &&
                userData['dateOfBirth'].isNotEmpty) {
              try {
                List<String> dateParts = userData['dateOfBirth'].split('/');
                if (dateParts.length == 3) {
                  _selectedDate = DateTime(
                    int.parse(dateParts[2]), // year
                    int.parse(dateParts[1]), // month
                    int.parse(dateParts[0]), // day
                  );
                }
              } catch (e) {
                print('Error parsing date: $e');
              }
            }

            _isLoadingData = false;
          });
        }
      }
    } catch (e) {
      print('Error loading user data: $e');
      _showSnackBar(
        'Error loading profile data',
        Colors.red[600]!,
        Icons.error,
      );
      setState(() {
        _isLoadingData = false;
      });
    }
  }

  Future<void> _selectDate(BuildContext context) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate:
          _selectedDate ??
          DateTime.now().subtract(const Duration(days: 6570)), // 18 years ago
      firstDate: DateTime(1950),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.light(
              primary: Colors.purple[700]!,
              onPrimary: Colors.white,
              onSurface: Colors.black,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null && picked != _selectedDate) {
      setState(() {
        _selectedDate = picked;
        _dobController.text = "${picked.day}/${picked.month}/${picked.year}";
      });
    }
  }

  void _updateProfile() async {
    // Prevent multiple submissions
    if (_isLoading) return;

    // Validate all fields
    if (_firstNameController.text.trim().isEmpty ||
        _lastNameController.text.trim().isEmpty ||
        _emailController.text.trim().isEmpty ||
        _selectedGender.isEmpty ||
        _dobController.text.trim().isEmpty) {
      _showSnackBar(
        'Please fill in all fields',
        Colors.orange[600]!,
        Icons.warning,
      );
      return;
    }

    // Validate email format
    if (!RegExp(
      r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$',
    ).hasMatch(_emailController.text.trim())) {
      _showSnackBar(
        'Please enter a valid email address',
        Colors.orange[600]!,
        Icons.warning,
      );
      return;
    }

    print('🚀 Starting profile update process...');
    setState(() {
      _isLoading = true;
    });

    try {
      User? currentUser = _auth.currentUser;
      if (currentUser == null) {
        throw Exception('No user logged in');
      }

      String fullName =
          '${_firstNameController.text.trim()} ${_lastNameController.text.trim()}';

      // Check if email has changed
      if (currentUser.email != _emailController.text.trim()) {
        print('📧 Updating email...');
        await currentUser.updateEmail(_emailController.text.trim());
        print('✅ Email updated successfully');
      }

      // Update display name
      await currentUser.updateDisplayName(fullName);
      print('👤 Display name updated: $fullName');

      // Update user data in Firestore
      print('💾 Updating user data in Firestore...');
      await _firestore.collection('users').doc(currentUser.uid).update({
        'firstName': _firstNameController.text.trim(),
        'lastName': _lastNameController.text.trim(),
        'email': _emailController.text.trim(),
        'gender': _selectedGender,
        'dateOfBirth': _dobController.text.trim(),
        'fullName': fullName,
        'updatedAt': FieldValue.serverTimestamp(),
      });
      print('✅ User data updated in Firestore');

      _showSnackBar(
        'Profile updated successfully!',
        Colors.green[600]!,
        Icons.check_circle,
      );

      // Navigate back after successful update
      if (mounted) {
        Navigator.pop(context);
      }
    } on FirebaseAuthException catch (e) {
      print('🔥 Firebase Auth Error: ${e.code} - ${e.message}');
      String errorMessage = _getAuthErrorMessage(e);
      _showSnackBar(errorMessage, Colors.red[600]!, Icons.error);
    } on FirebaseException catch (e) {
      print('🔥 Firebase Error: ${e.code} - ${e.message}');
      if (e.code == 'permission-denied') {
        _showSnackBar(
          'Permission denied. Please check your account settings.',
          Colors.red[600]!,
          Icons.error,
        );
      } else {
        _showSnackBar(
          'Database error: ${e.message}',
          Colors.red[600]!,
          Icons.error,
        );
      }
    } catch (e) {
      print('💥 General Error: $e');
      if (e.toString().contains('permission-denied')) {
        _showSnackBar(
          'Access denied. Please contact support.',
          Colors.red[600]!,
          Icons.error,
        );
      } else {
        _showSnackBar(
          'Something went wrong. Please try again.',
          Colors.red[600]!,
          Icons.error,
        );
      }
    } finally {
      print('🏁 Profile update process completed');
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  String _getAuthErrorMessage(FirebaseAuthException e) {
    switch (e.code) {
      case 'email-already-in-use':
        return 'This email is already registered. Please use a different email.';
      case 'invalid-email':
        return 'Please enter a valid email address.';
      case 'requires-recent-login':
        return 'Please log out and log back in before changing your email.';
      case 'network-request-failed':
        return 'Network error. Please check your internet connection.';
      default:
        return e.message ?? 'Something went wrong. Please try again.';
    }
  }

  void _showSnackBar(String message, Color backgroundColor, IconData icon) {
    if (!mounted) return;

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            Icon(icon, color: Colors.white),
            const SizedBox(width: 8),
            Expanded(child: Text(message)),
          ],
        ),
        backgroundColor: backgroundColor,
        duration: const Duration(seconds: 3),
        behavior: SnackBarBehavior.floating,
        margin: const EdgeInsets.all(16),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Edit Profile',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.purple[700],
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body:
          _isLoadingData
              ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Colors.purple[700]),
                    const SizedBox(height: 16),
                    Text(
                      'Loading profile data...',
                      style: TextStyle(color: Colors.grey[600]),
                    ),
                  ],
                ),
              )
              : Stack(
                children: [
                  // Dark gradient background
                  Container(
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          Color(0xFF1a237e), // Deep blue
                          Color(0xFF4a148c), // Deep purple
                          Color(0xFF311b92), // Indigo
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                    ),
                  ),
                  // Animated floating bubbles
                  AnimatedBuilder(
                    animation: _animation ?? const AlwaysStoppedAnimation(0.0),
                    builder: (context, child) {
                      double animationValue = _animation?.value ?? 0.0;
                      return Stack(
                        children: [
                          // Bubble 1
                          Positioned(
                            left: 50 + (100 * animationValue),
                            top: 100 + (50 * animationValue),
                            child: Container(
                              width: 80,
                              height: 80,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.white.withOpacity(0.1),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.2),
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                          // Bubble 2
                          Positioned(
                            right: 30 + (80 * (1 - animationValue)),
                            top: 200 + (30 * animationValue),
                            child: Container(
                              width: 120,
                              height: 120,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.purple.withOpacity(0.1),
                                border: Border.all(
                                  color: Colors.purple.withOpacity(0.2),
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                          // Bubble 3
                          Positioned(
                            left: 200 + (60 * animationValue),
                            bottom: 150 + (40 * (1 - animationValue)),
                            child: Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.blue.withOpacity(0.1),
                                border: Border.all(
                                  color: Colors.blue.withOpacity(0.2),
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                          // Bubble 4
                          Positioned(
                            right: 80 + (70 * animationValue),
                            bottom: 300 + (50 * animationValue),
                            child: Container(
                              width: 90,
                              height: 90,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.white.withOpacity(0.08),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.15),
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                          // Bubble 5
                          Positioned(
                            left: 30 + (90 * (1 - animationValue)),
                            bottom: 80 + (60 * animationValue),
                            child: Container(
                              width: 100,
                              height: 100,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.indigo.withOpacity(0.1),
                                border: Border.all(
                                  color: Colors.indigo.withOpacity(0.2),
                                  width: 1,
                                ),
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
                  // Centered edit form
                  Center(
                    child: SingleChildScrollView(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 30.0),
                        child: Container(
                          width: double.infinity,
                          constraints: const BoxConstraints(maxWidth: 400),
                          padding: const EdgeInsets.all(30),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.95),
                            borderRadius: BorderRadius.circular(20),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.2),
                                blurRadius: 20,
                                offset: const Offset(0, 10),
                              ),
                            ],
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              // Edit profile title
                              Text(
                                'Edit Profile',
                                style: TextStyle(
                                  fontSize: 28,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.grey[800],
                                ),
                              ),
                              const SizedBox(height: 10),
                              Text(
                                'Update your information',
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.grey[600],
                                ),
                              ),
                              const SizedBox(height: 30),
                              // First Name field
                              TextField(
                                controller: _firstNameController,
                                decoration: InputDecoration(
                                  labelText: 'First Name',
                                  labelStyle: TextStyle(
                                    color: Colors.grey[700],
                                  ),
                                  prefixIcon: Icon(
                                    Icons.person_outline,
                                    color: Colors.grey[600],
                                  ),
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                    ),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.purple[700]!,
                                      width: 2,
                                    ),
                                  ),
                                  filled: true,
                                  fillColor: Colors.grey[50],
                                ),
                                style: TextStyle(color: Colors.grey[800]),
                                textCapitalization: TextCapitalization.words,
                              ),
                              const SizedBox(height: 15),
                              // Last Name field
                              TextField(
                                controller: _lastNameController,
                                decoration: InputDecoration(
                                  labelText: 'Last Name',
                                  labelStyle: TextStyle(
                                    color: Colors.grey[700],
                                  ),
                                  prefixIcon: Icon(
                                    Icons.person_outline,
                                    color: Colors.grey[600],
                                  ),
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                    ),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.purple[700]!,
                                      width: 2,
                                    ),
                                  ),
                                  filled: true,
                                  fillColor: Colors.grey[50],
                                ),
                                style: TextStyle(color: Colors.grey[800]),
                                textCapitalization: TextCapitalization.words,
                              ),
                              const SizedBox(height: 15),
                              // Email field
                              TextField(
                                controller: _emailController,
                                decoration: InputDecoration(
                                  labelText: 'Email',
                                  labelStyle: TextStyle(
                                    color: Colors.grey[700],
                                  ),
                                  prefixIcon: Icon(
                                    Icons.email_outlined,
                                    color: Colors.grey[600],
                                  ),
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                    ),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.purple[700]!,
                                      width: 2,
                                    ),
                                  ),
                                  filled: true,
                                  fillColor: Colors.grey[50],
                                ),
                                style: TextStyle(color: Colors.grey[800]),
                                keyboardType: TextInputType.emailAddress,
                              ),
                              const SizedBox(height: 15),
                              // Gender selection
                              Container(
                                width: double.infinity,
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 8,
                                ),
                                decoration: BoxDecoration(
                                  border: Border.all(color: Colors.grey[300]!),
                                  borderRadius: BorderRadius.circular(12),
                                  color: Colors.grey[50],
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        Icon(
                                          Icons.people_outline,
                                          color: Colors.grey[600],
                                        ),
                                        const SizedBox(width: 8),
                                        Text(
                                          'Gender',
                                          style: TextStyle(
                                            color: Colors.grey[700],
                                            fontSize: 16,
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 8),
                                    Column(
                                      children: [
                                        Row(
                                          children: [
                                            Expanded(
                                              child: Row(
                                                children: [
                                                  Radio<String>(
                                                    value: 'Male',
                                                    groupValue: _selectedGender,
                                                    onChanged: (value) {
                                                      setState(() {
                                                        _selectedGender =
                                                            value!;
                                                      });
                                                    },
                                                    activeColor:
                                                        Colors.purple[700],
                                                    materialTapTargetSize:
                                                        MaterialTapTargetSize
                                                            .shrinkWrap,
                                                  ),
                                                  const Text('Male'),
                                                ],
                                              ),
                                            ),
                                            Expanded(
                                              child: Row(
                                                children: [
                                                  Radio<String>(
                                                    value: 'Female',
                                                    groupValue: _selectedGender,
                                                    onChanged: (value) {
                                                      setState(() {
                                                        _selectedGender =
                                                            value!;
                                                      });
                                                    },
                                                    activeColor:
                                                        Colors.purple[700],
                                                    materialTapTargetSize:
                                                        MaterialTapTargetSize
                                                            .shrinkWrap,
                                                  ),
                                                  const Text('Female'),
                                                ],
                                              ),
                                            ),
                                          ],
                                        ),
                                        Row(
                                          children: [
                                            Expanded(
                                              child: Row(
                                                children: [
                                                  Radio<String>(
                                                    value: 'Other',
                                                    groupValue: _selectedGender,
                                                    onChanged: (value) {
                                                      setState(() {
                                                        _selectedGender =
                                                            value!;
                                                      });
                                                    },
                                                    activeColor:
                                                        Colors.purple[700],
                                                    materialTapTargetSize:
                                                        MaterialTapTargetSize
                                                            .shrinkWrap,
                                                  ),
                                                  const Text('Other'),
                                                ],
                                              ),
                                            ),
                                            const Expanded(child: SizedBox()),
                                          ],
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(height: 15),
                              // Date of Birth field
                              TextField(
                                controller: _dobController,
                                decoration: InputDecoration(
                                  labelText: 'Date of Birth',
                                  labelStyle: TextStyle(
                                    color: Colors.grey[700],
                                  ),
                                  prefixIcon: Icon(
                                    Icons.calendar_today,
                                    color: Colors.grey[600],
                                  ),
                                  suffixIcon: Icon(
                                    Icons.arrow_drop_down,
                                    color: Colors.grey[600],
                                  ),
                                  border: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.grey[300]!,
                                    ),
                                  ),
                                  focusedBorder: OutlineInputBorder(
                                    borderRadius: BorderRadius.circular(12),
                                    borderSide: BorderSide(
                                      color: Colors.purple[700]!,
                                      width: 2,
                                    ),
                                  ),
                                  filled: true,
                                  fillColor: Colors.grey[50],
                                ),
                                style: TextStyle(color: Colors.grey[800]),
                                readOnly: true,
                                onTap: () => _selectDate(context),
                              ),
                              const SizedBox(height: 25),
                              // Update button with loading state
                              SizedBox(
                                width: double.infinity,
                                height: 50,
                                child: ElevatedButton(
                                  onPressed: _isLoading ? null : _updateProfile,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.purple[700],
                                    foregroundColor: Colors.white,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(12),
                                    ),
                                    elevation: 3,
                                  ),
                                  child:
                                      _isLoading
                                          ? const SizedBox(
                                            height: 20,
                                            width: 20,
                                            child: CircularProgressIndicator(
                                              color: Colors.white,
                                              strokeWidth: 2,
                                            ),
                                          )
                                          : const Text(
                                            'Update Profile',
                                            style: TextStyle(
                                              fontSize: 16,
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                ),
                              ),
                              const SizedBox(height: 20),
                              // Cancel button
                              TextButton(
                                onPressed: () {
                                  Navigator.pop(context);
                                },
                                child: Text(
                                  'Cancel',
                                  style: TextStyle(
                                    color: Colors.grey[600],
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                            ],
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
