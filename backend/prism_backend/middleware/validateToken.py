class ValidateTokenMiddleware:
    def __init__(self):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        # Import the google-auth modules to validate the token

        # If token is missing raise an exception

        # Refer to Exception Handling in Handling HTTP requests > Middleware > Exception Handling

        return response
