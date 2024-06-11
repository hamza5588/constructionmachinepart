from django.db import models

# Create your models here.


from django.db import models

class MachinePart(models.Model):
    machinemodel= models.CharField(max_length=100)
    parts = models.CharField(max_length=50)
    description = models.TextField()

    def __str__(self):
        return f"{self.part_no}: {self.description}"

