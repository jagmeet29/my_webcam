// Imports are equivalent to '#include' in C++ or 'import' in Python/Java.
import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

// The entry point of the application.
// Identical to 'public static void main(String[] args)' in Java or 'int main()' in C++.
void main() {
  runApp(const MyApp());
}

// Immutable base class. Think of this as a Java class with purely final properties.
class MyApp extends StatelessWidget {
  const MyApp({super.key}); // @USER= what is this line what is super.key

  @override // Tells the compiler we are overriding the parent class's method (Java equivalent).
  Widget build(BuildContext context) {
    // Returns the root structure of the app.
    // This is similar to defining the root HTML tag in a Flask Jinja template.
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: IPWebViewScreen(), // Sets the default screen to load.
    );
  }
}

// A widget that manages mutable state.
class IPWebViewScreen extends StatefulWidget {
  // 'final' means this variable can only be set once (like 'const' in C++ or 'final' in Java).
  final String ipAddress;

  // Constructor with a default parameter.
  // Equivalent to Python: def __init__(self, ipAddress="http://100.105.11.3:8090"):
  const IPWebViewScreen({
    super.key,
    this.ipAddress = "http://100.105.11.3:8090",
  });

  @override
  // Connects this widget to its mutable State class below.
  State<IPWebViewScreen> createState() => _IPWebViewScreenState(); // @USER= why this line of code looks likes this (syntax) what is happening here
}

// The State class where the logic happens.
// The underscore '_' makes the class private to this file (similar to Java's 'private' modifier).
// 'with WidgetsBindingObserver' is a Mixin (similar to C++ multiple inheritance) allowing the class to listen to app lifecycle events.
class _IPWebViewScreenState extends State<IPWebViewScreen>
    with WidgetsBindingObserver {
  // 'late' promises the compiler this will be initialized before it is used.
  // Conceptually similar to a pointer in C++ that gets assigned inside the constructor.
  late final WebViewController controller;

  @override
  void initState() {
    // super.initState() must be called first, just like calling super() inside a Python __init__.
    super.initState();

    // Registers this class to listen to system events (Observer Pattern).
    WidgetsBinding.instance.addObserver(this);

    // The '..' is Dart's cascade operator. It allows calling multiple methods on the same object instance.
    // In Python, this would look like:
    // controller = WebViewController()
    // controller.setJavaScriptMode(...)
    // controller.loadRequest(...)
    controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..loadRequest(Uri.parse(widget.ipAddress));
    // 'widget.ipAddress' accesses the variable from the parent IPWebViewScreen class.
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Triggers when the app goes to the background or comes back to the foreground.
    if (state == AppLifecycleState.resumed) {
      // Reloads the WebView (simulating a browser refresh) when the app is reopened.
      controller.reload();
    }
  }

  @override
  void dispose() {
    // Cleanup method to prevent memory leaks.
    // This is the direct equivalent of a C++ destructor: ~_IPWebViewScreenState()
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Scaffold provides the basic visual layout (like a blank canvas).
    return Scaffold(
      // SafeArea prevents the web view from drawing under the phone's camera notch or status bar.
      body: SafeArea(
        child: WebViewWidget(
          controller: controller,
        ), // Renders the actual web page.
      ),
    );
  }
}
