"""Serializers for the user API View."""

from django.contrib.auth import get_user_model

# serializers are just a way to convert from JSON file to python Objects and viceversa (it also makes sure is validated, correct, and secure)
# really useful for complex models such as Django models (database tables)
# to JSON file that we can send in a http repsonse
from rest_framework import serializers


# ModelSerializer is the automatic way to do it (simplest)
class UserSerializer(serializers.ModelSerializer):
    """Serializer for the user object."""

    # this is where we tell the serializer the model fields and any additional arguments we would like to pass to the serializer
    # the serializer nees to know wich model is representing thast why the
    # model is get_user_model()
    class Meta:
        """Metadata options for the user serializer or form.

        Attributes:
            model (Model): The user model retrieved via `get_user_model()`.
        """

        model = get_user_model()

        # these are the list of fields we want to make available trhough the serializer
        # we dont want to include is_admin because that would imply the user
        # will be able to change those values from its request (when they
        # create objects)
        fields = ["email", "password", "first_name", "last_name"]

        # a dictionary (wich is basically a JSON string) that allow us to provide extra metadata to the different fields to tell the django rest framework
        # things like do we want the field to be write only? read only? min-length? etc..
        # note it wont be any value returned from the api response for password (users cant read it)
        # 400 bad request is the result of a validation error that we will get
        # here if the password is not 5+ chars
        extra_kwargs = {"password": {"write_only": True, "min_length": 5}}

    # this function allow us to overrride the behavior the serializer does when you create new objects using that serializer
    # the default behavior just creates an object with wherever values are passed in verbatim
    # example: the password will be saved as clear text in the default
    # behavior but we want to encript it instead thats why we wont use the
    # default behavior
    def create(self, validated_data):
        """Create and return an user with encrypted password."""
        # we instead will use the create_user after we validated the data above
        # and that will encrypt the passowrd as we defined in the user manager
        return get_user_model().objects.create_user(**validated_data)
