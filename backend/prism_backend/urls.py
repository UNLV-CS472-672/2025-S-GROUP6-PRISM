"""
URL configuration for prism_backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path("django/admin/", admin.site.urls),
    # setting up the urls for the automatic API documentation
    path("api/schema/", SpectacularAPIView.as_view(), name="api-schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="api-schema"),
        name="api-docs",
    ),
    path("api/user/", include("users.urls")),
    path("api/course/", include("courses.urls")),
    path("api/assignment/", include("assignments.urls")),
    path("api/validate-token/", views.validate_token, name="validate_token"),
]

# from django.conf.urls import include, url
# from . import views

# urlpatterns = [
#    url(r'^django/', include([
#        url(r'^admin/', include(admin.site.urls) ),
#        url(r'^other/$', views.other)
#    ])),
# ]

# from django.conf.urls import patterns, url

# urlpatterns = patterns('',
#    (r'^django/admin/', include(admin.site.urls) ),
# )
#from django.conf.urls import patterns, url

#urlpatterns = patterns('',
#    (r'^django/admin/', include(admin.site.urls) ),
#)
