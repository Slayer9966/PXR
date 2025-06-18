import 'package:finalproject/screens/splash_screen.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'screens/signup_screen.dart';
import 'screens/login_screen.dart';
import 'screens/edit_my_profile_screen.dart';
import 'screens/my_profile_screen.dart';
import 'widgets/Navigation.dart'; // Make sure this contains class MainNavigation

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PXR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      initialRoute: '/splash',
      routes: {
        '/home': (context) => MainNavigation(),
        '/login': (context) => LoginPage(),
        '/my-profile': (context) => MyProfileScreen(),
        '/edit-profile': (context) => EditProfilePage(),
        '/signup': (context) => SignupPage(),
        '/splash': (context) => SplashScreen(),
      },
    );
  }
}
