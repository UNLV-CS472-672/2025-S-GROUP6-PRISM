# Generated by Django 5.1.5 on 2025-04-16 22:23

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("assignments", "0002_initial"),
        ("cheating", "0002_initial"),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name="submissionsimiliaritypairs",
            unique_together={("submission_id_1", "submission_id_2", "assignment")},
        ),
    ]
